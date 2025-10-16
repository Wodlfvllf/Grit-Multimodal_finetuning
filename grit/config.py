import torch


class GRITConfig:
	"""Configuration for GRIT training.

	Defaults are sensible for reproduction; override in scripts as needed.
	"""

	def __init__(self):
		self.model_id = "meta-llama/Llama-3.2-3B"
		self.batch_size = 8
		self.gradient_accumulation_steps = 4
		self.num_epochs = 3
		self.learning_rate = 2e-5
		self.precision = "bf16"
		self.max_length = 1024

		self.lora_rank = 16
		self.lora_alpha = 32
		self.lora_dropout = 0.0
		self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

		self.kfac_update_freq = 50
		self.kfac_damping = 0.001
		self.kfac_min_samples = 64  # minimum rank-space samples before attempting inversion
		self.reprojection_freq = 50
		self.reprojection_k = 8
		self.grit_cov_update_freq = 15

		# Reprojection mode: gate B-side onto G_cov basis (fallback to A_cov until well-sampled)
		self.use_two_sided_reprojection = True

		self.enable_rank_adaptation = True
		self.rank_adaptation_threshold = 0.99
		self.min_lora_rank = 4

		# Optional warmups (disabled by default)
		self.regularizer_warmup_steps = 0           # ramp lambda_kfac/lambda_reproj over first N steps
		self.reprojection_warmup_steps = 0          # skip reprojection entirely for first N steps
		self.rank_adaptation_start_step = 0         # delay adaptive k selection until this step
		# Natural-gradient warmup (skip NG until this global step)
		self.ng_warmup_steps = 0

		self.dataset_name = "tatsu-lab/alpaca"
		self.num_workers = 8
		self.pin_memory = True
		self.drop_last = True

		self.lambda_kfac = 1e-5
		self.lambda_reproj = 1e-4

		self.log_fisher_spectrum = True
		self.log_top_eigs = 8
		self.log_eig_heatmaps = True
		self.log_eigs_bar = True  # log eigenvalue spectrum as a single bar plot
		self.log_eig_heatmaps_modules = 6  # cap number of modules to log heatmaps for per run
		# Effective-rank logging controls
		self.log_eff_rank_on_inversion = False  # set True to log every inversion
		self.log_final_eff_rank = True         # log once at training end
		# K-FAC inversion device: 'cpu' (default) or 'cuda' (if available)
		self.kfac_inversion_device = 'cpu'


