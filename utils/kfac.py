
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

# class KFACStatistics:
#     """K-FAC statistics tracker for efficient second-order optimization"""
    
#     def __init__(self, input_dim: int, output_dim: int, momentum: float = 0.95, 
#                  device: str = "cuda", dtype=torch.float32):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.momentum = momentum
#         self.device = device
#         self.dtype = dtype
        
#         # Initialize covariance matrices
#         self.A = torch.eye(input_dim, device=device, dtype=dtype) * 1e-4
#         self.G = torch.eye(output_dim, device=device, dtype=dtype) * 1e-4
        
#         # Eigendecomposition cache
#         self.A_eigvals = None
#         self.A_eigvecs = None
#         self.G_eigvals = None
#         self.G_eigvecs = None
        
#     def update(self, inputs: torch.Tensor, grad_out: torch.Tensor):
#         """
#         Update K-FAC statistics with new batch.
#         Example shapes are for a transformer layer with:
#         - batch_size = 4
#         - sequence_length = 140
#         - input_dim = 1536
#         - output_dim = 256
#         """
#         # === Step 1: Get Feature Dimensions ===
#         # Get the feature dimension from each tensor individually. This is crucial for
#         # layers where input and output dimensions are different.
#         # input_dim will be 1536 for our example.
#         input_dim = inputs.shape[-1]
#         # output_dim will be 256 for our example.
#         output_dim = grad_out.shape[-1]

#         # === Step 2: Reshape Tensors for Statistical Calculation ===
#         # Reshape the 3D input tensor into a 2D matrix.
#         # This treats each token in the sequence as an independent sample.
#         # Shape goes from [4, 140, 1536] to [4 * 140, 1536] -> [560, 1536].
#         X = inputs.reshape(-1, input_dim).to(self.dtype)

#         # Reshape the 3D output gradient tensor into a 2D matrix.
#         # Shape goes from [4, 140, 256] to [4 * 140, 256] -> [560, 256].
#         G = grad_out.reshape(-1, output_dim).to(self.dtype)

#         # === Step 3: Compute Batch Covariance Matrices ===
#         # The number of samples for averaging is the first dimension of the reshaped matrix.
#         # effective_batch_size will be 560 in our example.
#         effective_batch_size = X.shape[0]

#         # Compute the input activation covariance matrix (A).
#         # X.T shape: [1536, 560]
#         # X shape:   [560, 1536]
#         # Resulting A_batch shape: [1536, 1536] ([input_dim, input_dim])
#         A_batch = (X.T @ X) / max(1, effective_batch_size)

#         # Compute the output gradient covariance matrix (G).
#         # G.T shape: [256, 560]
#         # G shape:   [560, 256]
#         # Resulting G_batch shape: [256, 256] ([output_dim, output_dim])
#         G_batch = (G.T @ G) / max(1, effective_batch_size)

#         # === Step 4: Update Running Averages with Momentum ===
#         # This is an element-wise operation, so the shapes of self.A and self.G do not change.
#         # self.A shape remains [1536, 1536].
#         self.A = self.momentum * self.A + (1.0 - self.momentum) * A_batch
#         # self.G shape remains [256, 256].
#         self.G = self.momentum * self.G + (1.0 - self.momentum) * G_batch
        
#     def compute_eigendecomp(self):
#         """Compute eigendecomposition of covariance matrices"""
#         self.A_eigvals, self.A_eigvecs = torch.linalg.eigh(self.A)
#         self.G_eigvals, self.G_eigvecs = torch.linalg.eigh(self.G)
        
#         # Clamp eigenvalues to avoid numerical issues
#         self.A_eigvals = torch.clamp(self.A_eigvals, min=1e-12)
#         self.G_eigvals = torch.clamp(self.G_eigvals, min=1e-12)


import torch

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
        
    def update(self, inputs: torch.Tensor, grad_out: torch.Tensor, debug=False):
        """
        Update K-FAC statistics with new batch.
        Includes a `debug` flag to print internal state.
        """
        # === Step 1: Get Feature Dimensions ===
        input_dim = inputs.shape[-1]
        output_dim = grad_out.shape[-1]

        # === Step 2: Reshape Tensors for Statistical Calculation ===
        X = inputs.reshape(-1, input_dim).to(self.dtype)
        G = grad_out.reshape(-1, output_dim).to(self.dtype)

        # === Step 3: Compute Batch Covariance Matrices ===
        effective_batch_size = X.shape[0]
        A_batch = (X.T @ X) / max(1, effective_batch_size)
        G_batch = (G.T @ G) / max(1, effective_batch_size)

        # === INTERNAL DEBUG PRINTS ===
        if debug:
            print(f"\n    --- KFAC Internal Trace ---")
            print(f"    - Input `grad_out` norm: {grad_out.norm().item():.6f}")
            print(f"    - Reshaped `G` norm:     {G.norm().item():.6f}")
            print(f"    - Calculated `G_batch` norm: {G_batch.norm().item():.6f}")
            print(f"    - `self.momentum`:       {self.momentum}")
            print(f"    - `self.G` norm BEFORE update: {self.G.norm().item():.6f}")

        # === Step 4: Update Running Averages with Momentum ===
        # This is the critical section we are investigating.
        self.A = self.momentum * self.A + (1.0 - self.momentum) * A_batch
        self.G = self.momentum * self.G + (1.0 - self.momentum) * G_batch
        
        if debug:
            print(f"    - `self.G` norm AFTER update:  {self.G.norm().item():.6f}")
            print(f"    ---------------------------\n")

    def compute_eigendecomp(self):
        """Compute eigendecomposition of covariance matrices"""
        self.A_eigvals, self.A_eigvecs = torch.linalg.eigh(self.A)
        self.G_eigvals, self.G_eigvecs = torch.linalg.eigh(self.G)
        
        self.A_eigvals = torch.clamp(self.A_eigvals, min=1e-12)
        self.G_eigvals = torch.clamp(self.G_eigvals, min=1e-12)