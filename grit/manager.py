import math
import gc
import torch
import wandb

from .autograd import KFACAutogradFunction


class GRITManager:
	"""Manager for instrumentation, covariance storage/inversion, preconditioning,
	and neural reprojection.
	"""

	def __init__(self, model, config, device):
		self.model = model
		self.config = config
		self.device = device
		self.global_step = 0
		self.backward_step = 0
		self.loss_history = []
		self.loss_history_capacity = 20
		self.last_kfac_update_step = 0
		self.last_reprojection_step = 0
		self.a_covs = {}
		self.g_covs = {}
		self.a_invs = {}
		self.g_invs = {}
		self.num_samples_a = {}
		self.num_samples_g = {}
		self.cov_update_tick = {}
		self.optimized_modules = []
		self.factors_are_ready = False
		# Cache latest Fisher top-k bases for reuse by other components (e.g., regularizer)
		self._last_Va_k = {}
		self._last_Vg_k = {}
		self._instrument_model()
		print("GRITManager: Initialization complete.")
		print(f"🔍 Optimizing {len(self.optimized_modules)} key LoRA modules.")
		print("💾 K-FAC covariances kept on-device; snapshot to CPU only at inversion.")

	def _wandb_log(self, data):
		try:
			if getattr(wandb, 'run', None) is not None:
				wandb.log(data, step=self.global_step)
		except Exception:
			pass

	def _compute_effective_rank(self, cov: torch.Tensor) -> float:
		"""Entropy-based effective rank: exp(H(p)), p = λ / trace(cov)."""
		try:
			with torch.no_grad():
				evals = torch.linalg.eigvalsh(cov.float())
				evals = torch.clamp(evals, min=0.0)
				total = float(evals.sum().item())
				if not math.isfinite(total) or total <= 0.0:
					return 0.0
				p = (evals / total).clamp_min(1e-12)
				entropy = float((-(p * torch.log(p)).sum()).item())
				r_eff = math.exp(entropy)
				n = int(cov.shape[0])
				if not math.isfinite(r_eff):
					return 0.0
				return float(max(0.0, min(n, r_eff)))
		except Exception:
			return 0.0

	def _instrument_model(self):
		from peft.tuners.lora import LoraLayer
		attention_modules = 0
		for name, module in self.model.named_modules():
			if isinstance(module, LoraLayer) and module.r['default'] > 0:
				module.lora_name = name
				module.grit_manager = self
				self.num_samples_a[module] = 0
				self.num_samples_g[module] = 0
				self.cov_update_tick[module] = 0
				module.original_forward = module.forward
				def new_forward(self, x=None, *args, **kwargs):
					if x is None:
						x = kwargs.get("hidden_states", None)
						if x is None and len(args) > 0:
							x = args[0]
					y = self.original_forward(x, *args, **kwargs)
					return KFACAutogradFunction.apply(self, y, x)
				module.forward = new_forward.__get__(module, LoraLayer)
				r = module.r['default']
				cov_device = self.device if torch.cuda.is_available() else torch.device('cpu')
				self.a_covs[module] = torch.zeros((r, r), device=cov_device, dtype=torch.float16)
				self.g_covs[module] = torch.zeros((r, r), device=cov_device, dtype=torch.float16)
				self.optimized_modules.append(module)
				attention_modules += 1
		print(f"Instrumented {attention_modules} LoRA modules with custom autograd.")
		print(f"Using r-dim ({self.config.lora_rank}x{self.config.lora_rank}) covariances.")
		# Keep covariances on-device during backprop to avoid frequent D2H copies; snapshots occur at inversion.

	def _get_adaptive_freq(self, base_freq, min_freq=1, max_freq=1000, window=20):
		if len(self.loss_history) < window:
			return base_freq
		recent_losses = self.loss_history[-window:]
		first_half = sum(recent_losses[:window//2]) / (window//2)
		second_half = sum(recent_losses[window//2:]) / (window//2)
		if second_half < first_half * 0.99:
			new_freq = int(base_freq * 1.5)
		else:
			new_freq = int(base_freq * 0.75)
		return max(min_freq, min(new_freq, max_freq))

	def step(self, loss=None):
		self.global_step += 1
		if loss is not None:
			self.loss_history.append(loss)
			if len(self.loss_history) > self.loss_history_capacity:
				self.loss_history = self.loss_history[-self.loss_history_capacity:]
		if len(self.loss_history) > 10:
			loss_variance = torch.tensor(self.loss_history).var().item()
			self.config.kfac_damping = max(1e-6, min(0.01, 0.001 + math.sqrt(loss_variance)))
			if self.global_step % 100 == 0:
				self._wandb_log({"adaptive_kfac_damping": self.config.kfac_damping})
		kfac_freq = self._get_adaptive_freq(self.config.kfac_update_freq)
		if self.global_step - self.last_kfac_update_step >= kfac_freq:
			self.update_and_invert_factors()
			self.last_kfac_update_step = self.global_step
		reproj_freq = self._get_adaptive_freq(self.config.reprojection_freq, min_freq=1, max_freq=2000)
		# Optional warmup to skip early reprojection entirely
		if self.global_step >= int(getattr(self.config, 'reprojection_warmup_steps', 0) or 0) and \
		   self.global_step - self.last_reprojection_step >= reproj_freq:
			self.neural_reprojection()
			self.last_reprojection_step = self.global_step

	def update_and_invert_factors(self):
		print(f"\nGRITManager: Inverting K-FAC factors at step {self.global_step}...")
		# Cache the scripted function to avoid re-scripting every inversion
		if not hasattr(self, "_scripted_invert_fn") or self._scripted_invert_fn is None:
			self._scripted_invert_fn = torch.jit.script(jit_invert_tensor_pair)
		scripted_invert_fn = self._scripted_invert_fn
		for module in self.optimized_modules:
			# Skip inversion if we have too few samples for stable stats
			min_samples = getattr(self.config, 'kfac_min_samples', 64)
			if self.num_samples_a.get(module, 0) < min_samples or self.num_samples_g.get(module, 0) < min_samples:
				print("K-FAC inversion deferred (insufficient samples).")
				continue
			# Snapshot to selected device for inversion ('cpu' by default)
			inv_device = str(getattr(self.config, 'kfac_inversion_device', 'cpu') or 'cpu')
			if inv_device == 'cuda' and torch.cuda.is_available():
				device_for_inv = torch.device('cuda')
			else:
				device_for_inv = torch.device('cpu')
			a_cov = self.a_covs[module].detach().to(device=device_for_inv, dtype=torch.float32)
			g_cov = self.g_covs[module].detach().to(device=device_for_inv, dtype=torch.float32)
			# Log entropy-based effective ranks per inversion only if enabled
			if getattr(self.config, 'log_eff_rank_on_inversion', False):
				try:
					r_eff_a = self._compute_effective_rank(a_cov)
					r_eff_g = self._compute_effective_rank(g_cov)
					self._wandb_log({
						f"Fisher/{module.lora_name}/eff_rank_a": float(r_eff_a),
						f"Fisher/{module.lora_name}/eff_rank_g": float(r_eff_g),
					})
				except Exception:
					pass
			# Optional eigenvalue logging (can be disabled to reduce CPU syncs)
			if getattr(self.config, 'log_fisher_spectrum', False) and getattr(self.config, 'log_top_eigs', 0) > 0:
				try:
					top_k = int(min(self.config.log_top_eigs, a_cov.shape[0]))
					if top_k > 0:
						evals_a = torch.linalg.eigvalsh(a_cov).flip(0)[:top_k].tolist()
						evals_g = torch.linalg.eigvalsh(g_cov).flip(0)[:top_k].tolist()
						log_vals = {}
						for i, v in enumerate(evals_a):
							log_vals[f"Fisher/{module.lora_name}/eig_a_{i}"] = float(v)
						for i, v in enumerate(evals_g):
							log_vals[f"Fisher/{module.lora_name}/eig_g_{i}"] = float(v)
						if log_vals:
							self._wandb_log(log_vals)
						# Single bar chart (optional)
						if getattr(self.config, 'log_eigs_bar', False):
							import matplotlib.pyplot as plt
							fig, ax = plt.subplots(1, 1, figsize=(6, 3))
							ax.bar(range(len(evals_a)), evals_a)
							ax.set_title(f"Top-{top_k} eigenvalues (A) | {module.lora_name}")
							ax.set_xlabel("Index")
							ax.set_ylabel("Eigenvalue")
							plt.tight_layout()
							self._wandb_log({f"Fisher/SpectrumBar/{module.lora_name}": wandb.Image(fig)})
							plt.close(fig)
				except Exception as _:
					pass
			a_inv, g_inv = scripted_invert_fn(a_cov=a_cov, g_cov=g_cov, kfac_damping=self.config.kfac_damping)
			if a_inv.numel() > 0 and g_inv.numel() > 0:
				# Keep inverses on CPU to avoid growing GPU memory when using GPU inversion
				self.a_invs[module] = a_inv.to('cpu')
				self.g_invs[module] = g_inv.to('cpu')
			else:
				print("K-FAC inversion failed for a module. Skipping.")
		self.factors_are_ready = True

	def log_final_effective_ranks(self):
		"""Compute and log final effective ranks once (CPU) for all modules."""
		if not getattr(self.config, 'log_final_eff_rank', True):
			return
		try:
			log = {}
			for module in self.optimized_modules:
				a_cov = self.a_covs[module].detach().to(device='cpu', dtype=torch.float32)
				g_cov = self.g_covs[module].detach().to(device='cpu', dtype=torch.float32)
				r_eff_a = self._compute_effective_rank(a_cov)
				r_eff_g = self._compute_effective_rank(g_cov)
				log[f"Fisher/{module.lora_name}/eff_rank_a"] = float(r_eff_a)
				log[f"Fisher/{module.lora_name}/eff_rank_g"] = float(r_eff_g)
			# Also aggregate statistics
			if log:
				self._wandb_log(log)
		except Exception:
			pass

	def log_final_raw_ranks(self):
		"""Log final raw integer ranks k per module using cached Fisher bases."""
		try:
			log = {}
			for module in self.optimized_modules:
				base_r = int(module.r['default']) if hasattr(module, 'r') and 'default' in module.r else 0
				Va = getattr(self, '_last_Va_k', {}).get(module, None)
				Vg = getattr(self, '_last_Vg_k', {}).get(module, None)
				k_a = int(Va.shape[1]) if Va is not None and Va.numel() > 0 else base_r
				k_b = int(Vg.shape[1]) if Vg is not None and Vg.numel() > 0 else k_a
				log[f"Fisher/{module.lora_name}/final_k_a"] = int(k_a)
				log[f"Fisher/{module.lora_name}/final_k_b"] = int(k_b)
				log[f"Fisher/{module.lora_name}/base_r"] = int(base_r)
			if log:
				self._wandb_log(log)
		except Exception:
			pass

	def precondition_gradients(self):
		if not self.factors_are_ready:
			return
		if self.global_step % self.config.kfac_update_freq == 0:
			print(f"\nGRITManager: Applying Natural Gradient preconditioner at step {self.global_step}...")
		with torch.no_grad():
			# Optional NG warmup: skip preconditioning for first N steps
			ng_warmup = int(getattr(self.config, 'ng_warmup_steps', 0) or 0)
			if self.global_step < ng_warmup:
				return
			dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
			# Accumulate a single scalar ||g_nat|| per step to avoid log spam
			_total_nat_norm_sq = torch.tensor(0.0, device=self.device)
			for module in self.optimized_modules:
				if module not in self.a_invs or module not in self.g_invs:
					continue
				lora_a = module.lora_A['default']
				lora_b = module.lora_B['default']
				if (lora_a is None or lora_b is None or lora_a.weight.grad is None or lora_b.weight.grad is None):
					continue
				# Compute NG in float32 for numerical stability, then cast back
				a_inv_f32 = self.a_invs[module].to(self.device, dtype=torch.float32)
				g_inv_f32 = self.g_invs[module].to(self.device, dtype=torch.float32)
				grad_b_f32 = lora_b.weight.grad.detach().to(torch.float32)
				# Correct sides for ΔW = B A and F ≈ G ⊗ A
				preconditioned_b_grad_f32 = grad_b_f32 @ g_inv_f32
				grad_a_f32 = lora_a.weight.grad.detach().to(torch.float32)
				r = module.r['default']
				if r > 0:
					grad_a_f32 = grad_a_f32.view(r, -1)
				preconditioned_a_grad_f32 = a_inv_f32 @ grad_a_f32
				# Update combined natural-gradient norm accumulator (fp32 for stability)
				_total_nat_norm_sq = _total_nat_norm_sq + preconditioned_a_grad_f32.pow(2).sum() + preconditioned_b_grad_f32.pow(2).sum()
				# Sanitize NaN/Inf and clamp extremes before casting back
				preconditioned_a_grad_f32 = torch.nan_to_num(preconditioned_a_grad_f32, nan=0.0, posinf=0.0, neginf=0.0)
				preconditioned_b_grad_f32 = torch.nan_to_num(preconditioned_b_grad_f32, nan=0.0, posinf=0.0, neginf=0.0)
				preconditioned_a_grad_f32 = torch.clamp(preconditioned_a_grad_f32, min=-1e6, max=1e6)
				preconditioned_b_grad_f32 = torch.clamp(preconditioned_b_grad_f32, min=-1e6, max=1e6)
				lora_a.weight.grad.copy_(preconditioned_a_grad_f32.to(lora_a.weight.grad.dtype))
				lora_b.weight.grad.copy_(preconditioned_b_grad_f32.to(lora_b.weight.grad.dtype))
				del a_inv_f32, g_inv_f32, grad_a_f32, grad_b_f32, preconditioned_a_grad_f32, preconditioned_b_grad_f32
			# Lightweight, occasional logging (every N steps)
			try:
				log_every = int(getattr(self.config, 'log_nat_norm_every', 50) or 50)
				if log_every > 0 and (self.global_step % log_every == 0):
					total = torch.clamp(_total_nat_norm_sq, min=0.0)
					g_nat_norm = float((torch.sqrt(total) if torch.isfinite(total) else torch.tensor(0.0, device=total.device)).detach().cpu())
					cap = getattr(self, '_last_grad_clip_cap', None)
					if cap is not None:
						print(f"[GRIT] ||g_nat|| = {g_nat_norm:.4f} (cap {cap}) (step {self.global_step})")
					else:
						print(f"[GRIT] ||g_nat|| = {g_nat_norm:.4f} (step {self.global_step})")
					self._wandb_log({"GRIT/g_nat_norm": g_nat_norm})
			except Exception:
				pass

	def neural_reprojection(self):
		print(f"\nGRITManager: Neural reprojection at step {self.global_step}...")
		initial_params = 0
		final_params = 0
		log_dict = {}
	
		with torch.no_grad():
			for module in self.optimized_modules:
				if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
					initial_params += module.r['default'] * (module.in_features + module.out_features)

			for module in self.optimized_modules:
				try:
					lora_a = module.lora_A['default']
					lora_b = module.lora_B['default']
					if lora_a is None or lora_b is None:
						continue

					A = lora_a.weight.data.float()
					B = lora_b.weight.data.float()
					r = A.shape[0]
					k = r
					M = self.a_covs[module].float()
					if torch.isnan(M).any() or torch.isinf(M).any():
						final_params += r * (module.in_features + module.out_features)
						continue
					# --- Sample gating to avoid unstable early projection ---
					min_samples = int(getattr(self.config, 'kfac_min_samples', 64) or 64)
					if self.num_samples_a.get(module, 0) < min_samples:
						# Skip reprojection for this module until A-cov is sufficiently sampled
						final_params += r * (module.in_features + module.out_features)
						log_dict[f"reprojection_skips/a_samples/{module.lora_name}"] = log_dict.get(f"reprojection_skips/a_samples/{module.lora_name}", 0) + 1
						continue

					if self.config.enable_rank_adaptation and r > self.config.min_lora_rank and \
					   self.global_step >= int(getattr(self.config, 'rank_adaptation_start_step', 0) or 0):
						evals_a, V_a = torch.linalg.eigh(self.a_covs[module].float())
						order_a = torch.argsort(evals_a, descending=True)
						V_a = V_a[:, order_a]
						evals_a = evals_a[order_a]
						total_energy = torch.sum(evals_a)
						if total_energy > 1e-6:
							cumulative_energy = torch.cumsum(evals_a, dim=0) / total_energy
							k = (cumulative_energy < self.config.rank_adaptation_threshold).sum().item() + 1
						k = max(k, self.config.min_lora_rank)
						k = min(k, r)
						V_a_k = V_a[:, :k]
						# Prefer g_cov eigenbasis for B when sufficiently sampled
						try:
							g_cov_m = self.g_covs.get(module, None)
							if getattr(self.config, 'use_two_sided_reprojection', False) and g_cov_m is not None and self.num_samples_g.get(module, 0) >= min_samples:
								evals_g, V_g = torch.linalg.eigh(g_cov_m.float())
								order_g = torch.argsort(evals_g, descending=True)
								V_g_k = V_g[:, order_g][:, :k]
							else:
								V_g_k = V_a_k
						except Exception:
							V_g_k = V_a_k
					else:
						k = min(self.config.reprojection_k, r)
						evals_a, V_a = torch.linalg.eigh(self.a_covs[module].float())
						order_a = torch.argsort(evals_a, descending=True)
						V_a_k = V_a[:, order_a][:, :k]
						# Prefer g_cov eigenbasis for B when sufficiently sampled
						try:
							g_cov_m = self.g_covs.get(module, None)
							if getattr(self.config, 'use_two_sided_reprojection', False) and g_cov_m is not None and self.num_samples_g.get(module, 0) >= min_samples:
								evals_g, V_g = torch.linalg.eigh(g_cov_m.float())
								order_g = torch.argsort(evals_g, descending=True)
								V_g_k = V_g[:, order_g][:, :k]
							else:
								V_g_k = V_a_k
						except Exception:
							V_g_k = V_a_k
					# Cache latest Fisher bases on CPU for reuse
					self._last_Va_k[module] = V_a_k.detach().cpu()
					self._last_Vg_k[module] = V_g_k.detach().cpu()

					# --- Heatmap plotting logic for each module ---
					if getattr(self.config, "log_eig_heatmaps", False):
						try:
							import matplotlib.pyplot as plt
							cap = int(getattr(self.config, 'log_eig_heatmaps_modules', 6) or 0)
							logged = getattr(self, '_heatmap_logged_count', 0)
							if cap > 0 and logged >= cap:
								pass
							else:
								Va_k = V_a_k.detach().cpu().float()
								A_proj = (Va_k.T @ A.cpu())
								energy = (A_proj.pow(2).sum(dim=1))
								
								fig, ax = plt.subplots(1, 2, figsize=(8, 3))
								ax[0].imshow(Va_k.abs().numpy(), aspect='auto', cmap='coolwarm')
								ax[0].set_title(f"V_a_k | {module.lora_name}")
								ax[0].set_xlabel("Eigenvector Index")
								ax[0].set_ylabel("Dimension Index")
								
								ax[1].bar(range(len(energy)), energy.numpy())
								ax[1].set_title("Update energy")
								ax[1].set_xlabel("Eigenvector Index")
								ax[1].set_ylabel("Squared Update Norm")
								
								plt.tight_layout()
								self._wandb_log({f"GRIT/Heatmaps/{module.lora_name}": wandb.Image(fig)})
								plt.close(fig)
								setattr(self, '_heatmap_logged_count', logged + 1)
						except Exception as e:
							print(f"Failed to log heatmap for {module.lora_name}: {e}")

					# --- Reprojection ---
					V_a_k_d = V_a_k.to(device=A.device, dtype=A.dtype, non_blocking=True)
					V_g_k_d = V_g_k.to(device=B.device, dtype=A.dtype, non_blocking=True)
					A_proj = V_a_k_d.T @ A
					B_proj = B @ V_g_k_d
					A_new = V_a_k_d @ A_proj
					B_new = B_proj @ V_g_k_d.T
					lora_a.weight.data.copy_(A_new.to(lora_a.weight.dtype))
					lora_b.weight.data.copy_(B_new.to(lora_b.weight.dtype))

					if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
						final_params += k * (module.in_features + module.out_features)

				except Exception as e:
					print(f"Error during reprojection for {module.lora_name}: {e}")
					if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
						final_params += module.r['default'] * (module.in_features + module.out_features)
		
		# --- Logging ---
		if initial_params > 0:
			param_reduction = initial_params - final_params
			reduction_percent = (param_reduction / initial_params) * 100
			print("Neural reprojection completed. Effective parameter count reduced.")
			print(f"    - Initial GRIT params: {initial_params:,}")
			print(f"    - Final GRIT params:    {final_params:,}")
			print(f"    - Reduction:            {param_reduction:,} ({reduction_percent:.2f}%)")
			log_dict.update({
				"Parameters/GRIT Initial Params": initial_params,
				"Parameters/GRIT Final Params": final_params,
				"Parameters/GRIT Param Reduction (%)": reduction_percent,
			})
		if log_dict:
			self._wandb_log(log_dict)
		gc.collect()
		torch.cuda.synchronize()

# NOTE: This function lives outside GRITManager on purpose.
# - It is a pure, stateless linear algebra kernel (only depends on inputs).
# - JIT/tracing works best on isolated math routines, not large stateful classes.
# - Keeps GRITManager focused on "when/why" to invert, not "how" to invert.
# - Easier to reuse (other K-FAC / preconditioning code) and to debug independently.
def jit_invert_tensor_pair(a_cov: torch.Tensor, g_cov: torch.Tensor, kfac_damping: float):
	with torch.no_grad():
		# Robust, per-matrix damping using Cholesky info (no eigen-decompositions)
		base_damping = float(max(kfac_damping, 1e-6))
		# Symmetrize to improve numerical stability
		a_cov_f = a_cov.float(); a_cov_f = 0.5 * (a_cov_f + a_cov_f.T)
		g_cov_f = g_cov.float(); g_cov_f = 0.5 * (g_cov_f + g_cov_f.T)
		n_a = int(a_cov_f.shape[0]); n_g = int(g_cov_f.shape[0])
		dev = a_cov_f.device
		I_a = torch.eye(n_a, device=dev); I_g = torch.eye(n_g, device=dev)
		# Deterministic damping escalation sequence
		scales = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
		for s in scales:
			damp_a = float(base_damping * s)
			damp_g = float(base_damping * s)
			L_a, info_a = torch.linalg.cholesky_ex(a_cov_f + damp_a * I_a)
			L_g, info_g = torch.linalg.cholesky_ex(g_cov_f + damp_g * I_g)
			if int(info_a.item()) == 0 and int(info_g.item()) == 0:
				a_inv = torch.cholesky_inverse(L_a).float()
				g_inv = torch.cholesky_inverse(L_g).float()
				return a_inv, g_inv
		# If all attempts fail, return empty tensors to skip this update
		return torch.empty(0, device=dev), torch.empty(0, device=dev)
