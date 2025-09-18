
from ..config import GRITConfig
from typing import List, Dict
from .util import *

def damp_and_invert(mat: torch.Tensor, damping: float) -> torch.Tensor:
    """Add damping to diagonal and invert matrix"""
    mat = mat + torch.eye(mat.shape[0], device=mat.device, dtype=mat.dtype) * damping
    try:
        return torch.linalg.inv(mat)
    except:
        # Fallback to pseudo-inverse if singular
        return torch.linalg.pinv(mat)

class KFACStatistics:
    """K-FAC statistics tracker for efficient second-order optimization"""
    
    def __init__(self, input_dim: int, output_dim: int, momentum: float = 0.95, 
                 device: str = "cuda", dtype=torch.float32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.momentum = momentum
        self.device = device
        self.dtype = dtype
        
        # Initialize covariance matrices
        self.A = torch.eye(input_dim, device=device, dtype=dtype) * 1e-4
        self.G = torch.eye(output_dim, device=device, dtype=dtype) * 1e-4
        
        # Eigendecomposition cache
        self.A_eigvals = None
        self.A_eigvecs = None
        self.G_eigvals = None
        self.G_eigvecs = None
        
    def update(self, inputs: torch.Tensor, grad_out: torch.Tensor):
        """Update K-FAC statistics with new batch"""
        b = inputs.shape[-1]

        # logger.info(f"KFAC update - inputs shape: {inputs.shape}, grad_out shape: {grad_out.shape}, "
            # f"inputs mem: {inputs.numel() * 4 / 1e9:.2f} GB")

        # Convert to correct dtype and reshape
        X = inputs.reshape(-1, b).to(self.dtype)
        G = grad_out.reshape(-1, self.output_dim).to(self.dtype)
        
        # logger.info(f"KFAC update - X shape: {X.shape}, G shape: {G.shape}, "
        #     f"X mem: {X.numel() * 4 / 1e9:.2f} GB")

        # Compute batch covariances
        A_batch = (X.T @ X) / max(1, b)
        G_batch = (G.T @ G) / max(1, b)
        
        # Update with momentum
        self.A = self.momentum * self.A + (1.0 - self.momentum) * A_batch
        self.G = self.momentum * self.G + (1.0 - self.momentum) * G_batch
        
    def compute_eigendecomp(self):
        """Compute eigendecomposition of covariance matrices"""
        self.A_eigvals, self.A_eigvecs = torch.linalg.eigh(self.A)
        self.G_eigvals, self.G_eigvecs = torch.linalg.eigh(self.G)
        
        # Clamp eigenvalues to avoid numerical issues
        self.A_eigvals = torch.clamp(self.A_eigvals, min=1e-12)
        self.G_eigvals = torch.clamp(self.G_eigvals, min=1e-12)