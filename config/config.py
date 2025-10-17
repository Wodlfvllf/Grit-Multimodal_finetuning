import torch
from typing import List
from dataclasses import dataclass, field

@dataclass
class GRITConfig:
    """Configuration for GRIT fine-tuning"""
    # Model configuration
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    precision: str = "bf16"

    # LoRA specific parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # K-FAC parameters
    kfac_update_freq: int = 50
    kfac_damping: float = 0.001
    kfac_min_samples: int = 64
    grit_cov_update_freq: int = 15

    # Reprojection parameters
    reprojection_freq: int = 50
    reprojection_k: int = 8
    use_two_sided_reprojection: bool = True

    # Rank adaptation
    enable_rank_adaptation: bool = True
    rank_adaptation_threshold: float = 0.99
    min_lora_rank: int = 4

    # Warmups
    regularizer_warmup_steps: int = 0
    reprojection_warmup_steps: int = 0
    rank_adaptation_start_step: int = 0
    ng_warmup_steps: int = 0

    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    num_epochs: int = 3
    max_grad_norm: float = 1.0

    # Data parameters
    max_length: int = 1024

    # System parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 1
    pin_memory: bool = True
    drop_last: bool = True

    # Regularization parameters
    lambda_kfac: float = 1e-5
    lambda_reproj: float = 1e-4

    # Logging
    log_fisher_spectrum: bool = True
    log_top_eigs: int = 8
    log_eig_heatmaps: bool = True
    log_eigs_bar: bool = True
    log_eig_heatmaps_modules: int = 6
    log_eff_rank_on_inversion: bool = False
    log_final_eff_rank: bool = True

    # K-FAC inversion device
    kfac_inversion_device: str = 'cpu'