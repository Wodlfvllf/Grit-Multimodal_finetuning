import gc
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from transformers import Seq2SeqTrainer, TrainerCallback

from ..grit.optim import GritOptimizer


class GritCallback(TrainerCallback):
    def __init__(self, grit_manager):
        self.grit_manager = grit_manager

    def on_step_end(self, args, state, control, **kwargs):
        last_loss = state.log_history[-1].get("loss") if state.log_history else None
        self.grit_manager.step(loss=last_loss)

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self.grit_manager.log_final_effective_ranks()
        except Exception:
            pass
        try:
            self.grit_manager.log_final_raw_ranks()
        except Exception:
            pass


class GritTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass that injects GRIT preconditioning and regularizers."""

    def __init__(self, grit_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grit_manager = grit_manager
        try:
            setattr(self.grit_manager, "_last_grad_clip_cap", float(getattr(self.args, "max_grad_norm", 0.0)))
        except Exception:
            pass
        print("GritTrainer: Initialized with GRIT implementation.")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        if self.optimizer is not None:
            print("🎁 Wrapping the optimizer with GRIT preconditioning logic.")
            self.optimizer = GritOptimizer(self.optimizer, self.grit_manager)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        lambda_k = getattr(self.grit_manager.config, "lambda_kfac", 0.0)
        lambda_r = getattr(self.grit_manager.config, "lambda_reproj", 0.0)
        warmup_steps = int(getattr(self.grit_manager.config, "regularizer_warmup_steps", 0) or 0)
        if warmup_steps > 0:
            prog = min(1.0, max(0.0, self.grit_manager.global_step / float(warmup_steps)))
            lambda_k = float(lambda_k) * prog
            lambda_r = float(lambda_r) * prog
        curv_reg = torch.tensor(0.0, device=base_loss.device)
        reproj_reg = torch.tensor(0.0, device=base_loss.device)
        for module in getattr(self.grit_manager, "optimized_modules", []):
            lora_a = module.lora_A['default'] if 'default' in module.lora_A else None
            lora_b = module.lora_B['default'] if 'default' in module.lora_B else None
            if lora_a is None or lora_b is None:
                continue
            A = lora_a.weight
            B = lora_b.weight
            a_cov = self.grit_manager.a_covs.get(module, None)
            g_cov = self.grit_manager.g_covs.get(module, None)
            if a_cov is not None:
                A_f = A.float()
                a_cov_f = a_cov.to(A_f.device, dtype=A_f.dtype)
                curv_reg = curv_reg + ((a_cov_f @ A_f) * A_f).sum()
            if g_cov is not None:
                B_f = B.float()
                g_cov_f = g_cov.to(B_f.device, dtype=B_f.dtype)
                curv_reg = curv_reg + ((B_f @ g_cov_f) * B_f).sum()
            Va_cache = getattr(self.grit_manager, "_last_Va_k", {})
            Vg_cache = getattr(self.grit_manager, "_last_Vg_k", {})
            V_a_k = Va_cache.get(module, None)
            V_g_k = Vg_cache.get(module, None)
            if V_a_k is None and a_cov is not None:
                try:
                    evals, evecs = torch.linalg.eigh(a_cov.float())
                    order = torch.argsort(evals, descending=True)
                    evecs = evecs[:, order]
                    total = torch.clamp(evals.sum(), min=1e-8)
                    threshold = float(self.grit_manager.config.rank_adaptation_threshold)
                    if self.grit_manager.global_step < int(getattr(self.grit_manager.config, 'rank_adaptation_start_step', 0) or 0):
                        k = min(self.grit_manager.config.reprojection_k, A.shape[0])
                    else:
                        k = int((torch.cumsum(evals, 0) / total < threshold).sum().item()) + 1
                    k = max(min(k, A.shape[0]), self.grit_manager.config.min_lora_rank)
                    V_a_k = evecs[:, :k]
                    if V_g_k is None:
                        try:
                            if (
                                getattr(self.grit_manager.config, 'use_two_sided_reprojection', False)
                                and g_cov is not None
                                and self.grit_manager.num_samples_g.get(module, 0) >= int(getattr(self.grit_manager.config, 'kfac_min_samples', 64) or 64)
                            ):
                                evals_g, V_gmat = torch.linalg.eigh(g_cov.float())
                                order_g = torch.argsort(evals_g, descending=True)
                                V_g_k = V_gmat[:, order_g][:, :k]
                            else:
                                V_g_k = V_a_k
                        except Exception:
                            V_g_k = V_a_k
                except Exception:
                    V_a_k = None
                    V_g_k = None
            if V_a_k is not None and V_a_k.numel() > 0:
                A_f = A.float(); B_f = B.float()
                device, dtype = A_f.device, A_f.dtype
                V_a_k_d = V_a_k.to(device=device, dtype=dtype, non_blocking=True)
                V_g_k_d = (V_g_k if V_g_k is not None else V_a_k).to(device=device, dtype=dtype, non_blocking=True)
                P_a = V_a_k_d @ V_a_k_d.T
                I_a = torch.eye(P_a.shape[0], device=device, dtype=dtype)
                P_g = V_g_k_d @ V_g_k_d.T
                I_g = torch.eye(P_g.shape[0], device=device, dtype=dtype)
                A_res = (I_a - P_a) @ A_f
                B_res = B_f @ (I_g - P_g)
                reproj_reg = reproj_reg + (A_res.pow(2).sum() + B_res.pow(2).sum())
        loss = base_loss + lambda_k * curv_reg + lambda_r * reproj_reg
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def optimizer_step(self, *args, **kwargs):
        setattr(self.grit_manager, "_preconditioned", False)
        if getattr(self.grit_manager, "factors_are_ready", False):
            self.grit_manager.precondition_gradients()
            setattr(self.grit_manager, "_preconditioned", True)
        return super().optimizer_step(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        print("\nð§¹ Clearing VRAM before evaluation...")
        gc.collect()
        torch.cuda.empty_cache()
        return super().evaluate(*args, **kwargs)