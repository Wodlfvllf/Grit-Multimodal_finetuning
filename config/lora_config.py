
# ============================================================================
# LoRA Configuration
# ============================================================================
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoRAConfig:
    """Configuration for standard LoRA fine-tuning"""
    
    # Model settings
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    
    # LoRA-specific parameters
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = None  # Will be set in __post_init__
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4
    
    # Data settings
    data_path: str = "/path/to/your/dataset"
    max_length: int = 512
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Qwen2-VL
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
            ]