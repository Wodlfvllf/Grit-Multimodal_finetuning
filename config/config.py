import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import logging
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import warnings

@dataclass
class GRITConfig:
    """Configuration for GRIT fine-tuning"""
    # Model configuration
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    
    # GRIT specific parameters
    rank: int = 32  # Low-rank dimension
    alpha: float = 16.0  # LoRA scaling factor
    dropout: float = 0.1
    target_modules: List[str] = None  # Will be set based on model architecture
    
    # K-FAC parameters
    kfac_update_freq: int = 10  # Update K-FAC statistics every N steps
    kfac_damping: float = 0.1   # Damping factor for K-FAC
    kfac_momentum: float = 0.5  # Momentum for K-FAC statistics
    
    # Neural reprojection parameters
    reprojection_rank: int = 4  # Number of top eigenvectors to use
    reprojection_freq: int = 50  # Recompute eigenvectors every N steps
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Data parameters
    image_size: int = 64
    max_length: int = 512
    
    # System parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False
    num_workers: int = 4
    
    # Regularization parameters
    lambda_curvature: float = 0.01
    lambda_reprojection: float = 0.01
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            # self.target_modules = ["q_proj", "k_proj"]
            