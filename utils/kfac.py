
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
        b = inputs.shape[0]
        
        # Convert to correct dtype and reshape
        X = inputs.reshape(b, -1).to(self.dtype)
        G = grad_out.reshape(b, -1).to(self.dtype)
        
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