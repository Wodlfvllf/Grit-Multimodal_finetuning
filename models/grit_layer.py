# ============================================================================
# GRIT Layer Implementation
# ============================================================================
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
from ..utils import KFACStatistics, damp_and_invert
from ..config import GRITConfig
from ..data import VQADataset


class LinearWithGRIT(nn.Module):
    """Linear layer with GRIT adaptation"""
    
    def __init__(self, orig_linear: nn.Linear, cfg):
        super().__init__()
        self.cfg = cfg
        self.base = orig_linear
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        
        self.r = self.cfg.rank
        self.alpha = self.cfg.alpha
        self.scaling = self.alpha / max(1, self.r)
        self.dropout_p = self.cfg.dropout
        
        device = next(orig_linear.parameters()).device
        dtype = next(orig_linear.parameters()).dtype
        
        self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=device, dtype=dtype))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.kfac = KFACStatistics(
            self.in_features, 
            self.out_features, 
            momentum=self.cfg.kfac_momentum, 
            device=device, 
            dtype=torch.float32
        )
        
        self.reprojection_rank = self.cfg.reprojection_rank
        
        self._saved_input = None
        self._saved_grad_out = None
        
        self.base.register_forward_hook(self._forward_hook)
        self.base.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, inputs, output):
        self._saved_input = inputs[0].detach()
        
    def _backward_hook(self, module, grad_input, grad_output):
        self._saved_grad_out = grad_output[0].detach()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_d = self.dropout(x)
        lora_mid = F.linear(x_d, self.lora_A)
        lora_out = F.linear(lora_mid, self.lora_B)
        return base_out + lora_out * self.scaling

    # =============================================================================
    # NEW SEPARATE FUNCTION FOR NEURAL REPROJECTION
    # =============================================================================
    def _apply_reprojection(self, grad_A_pre: torch.Tensor, grad_B_pre: torch.Tensor, debug: bool = False):
        """
        Projects the preconditioned gradients onto the top-k eigenspace of the
        K-FAC matrices.
        """
        if debug:
            print(f"\n[Step 6] Applying Neural Reprojection")

        # Compute eigendecomposition (this can be expensive)
        self.kfac.compute_eigendecomp()

        # Ensure eigenvectors are available
        if self.kfac.G_eigvecs is None or self.kfac.A_eigvecs is None:
            if debug:
                print("  - [SKIPPING] Eigendecomposition not available.")
            return grad_A_pre, grad_B_pre
            
        k = min(self.cfg.reprojection_rank, self.kfac.G_eigvecs.shape[1], self.kfac.A_eigvecs.shape[1])
        
        if debug:
            print(f"  - Reprojecting onto rank k={k}")
        
        if k > 0:
            top_g = self.kfac.G_eigvecs[:, -k:]  # (out, k)
            top_a = self.kfac.A_eigvecs[:, -k:]  # (in, k)
            
            # Project grad_B_pre onto the subspace spanned by top_g
            grad_B_final = top_g @ (top_g.T @ grad_B_pre)

            # Project grad_A_pre onto the subspace spanned by top_a
            grad_A_final = (grad_A_pre @ top_a) @ top_a.T
            
            if debug:
                norm_A_final = grad_A_final.norm().item()
                norm_B_final = grad_B_final.norm().item()
                print(f"  - Final grad norm A after reprojection: {norm_A_final:.6f}")
                print(f"  - Final grad norm B after reprojection: {norm_B_final:.6f}")
            
            return grad_A_final, grad_B_final
        
        # If k=0, return original preconditioned gradients
        return grad_A_pre, grad_B_pre

    def update_kfac_and_compute_preconditioned_grads(self, current_step: int = 0, debug: bool = False):
        """
        Update K-FAC statistics and precondition gradients.
        Now takes a `current_step` argument to control reprojection frequency.
        """
        # ... (Your existing debug prints and Steps 1-2 for K-FAC update remain the same) ...
        # [The following is a condensed version of your existing code for clarity]

        if self._saved_input is None or self._saved_grad_out is None: return
        self.kfac.update(self._saved_input, self._saved_grad_out, debug=debug)
        if self.lora_A.grad is None or self.lora_B.grad is None: return
        
        grad_A_auto = self.lora_A.grad.clone().to(torch.float32)
        grad_B_auto = self.lora_B.grad.clone().to(torch.float32)

        A_inv = damp_and_invert(self.kfac.A.to(torch.float32), self.cfg.kfac_damping)
        G_inv = damp_and_invert(self.kfac.G.to(torch.float32), self.cfg.kfac_damping)
        
        grad_B_pre = G_inv @ grad_B_auto
        grad_A_pre = grad_A_auto @ A_inv
        
        # Initialize final gradients as the preconditioned ones
        grad_A_final = grad_A_pre
        grad_B_final = grad_B_pre

        # =============================================================================
        # MODIFIED LOGIC TO CONDITIONALLY CALL REPROJECTION
        # =============================================================================
        # Check if reprojection is enabled AND if it's the right step to perform it
        if self.reprojection_rank > 0 and current_step > 0 and current_step % self.cfg.project_gradients_step == 0:
            grad_A_final, grad_B_final = self._apply_reprojection(grad_A_pre, grad_B_pre, debug)
        
        # === Final Update ===
        with torch.no_grad():
            self.lora_A.grad.copy_(grad_A_final.to(self.lora_A.dtype))
            self.lora_B.grad.copy_(grad_B_final.to(self.lora_B.dtype))

        # === Cleanup ===
        self._saved_input = None
        self._saved_grad_out = None
        
        if debug:
            print(f"\n[Step 7] Copied final gradients back to .grad attribute.")
            print(f"{'='*22} END TRACE {'='*23}\n")
    
    def merge_to_base(self):
        """Merge LoRA weights into base model"""
        with torch.no_grad():
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.data += delta_W
    
    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict for adapter parameters"""
        return {
            'lora_A': self.lora_A.data,
            'lora_B': self.lora_B.data
        }
    
    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load adapter parameters from state dict"""
        self.lora_A.data = state_dict['lora_A']
        self.lora_B.data = state_dict['lora_B']





# class LinearWithGRIT(nn.Module):
#     """Linear layer with GRIT adaptation"""
    
#     def __init__(self, orig_linear: nn.Linear, cfg):
#         super().__init__()
#         self.base = orig_linear  # Keep original linear frozen
#         self.in_features = orig_linear.in_features
#         self.out_features = orig_linear.out_features
        
#         # GRIT/LoRA parameters
#         self.r = cfg.rank
#         self.alpha = cfg.alpha
#         self.scaling = self.alpha / max(1, self.r)
#         self.dropout_p = cfg.dropout
        
#         # Initialize LoRA parameters
#         device = next(orig_linear.parameters()).device
#         dtype = next(orig_linear.parameters()).dtype
        
#         self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features, device=device, dtype=dtype))
#         self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=device, dtype=dtype))
        
#         # Initialize using Kaiming uniform
#         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B)
        
#         self.dropout = nn.Dropout(self.dropout_p)
        
#         # K-FAC statistics
#         self.kfac = KFACStatistics(
#             self.in_features, 
#             self.out_features, 
#             momentum=cfg.kfac_momentum, 
#             device=device, 
#             dtype=torch.float32
#         )
        
#         # Reprojection matrix (will be computed during training)
#         self.reprojection_matrix = None
#         self.reprojection_rank = cfg.reprojection_rank
        
#         # Saved tensors for backward pass
#         self._saved_input = None
#         self._saved_grad_out = None
        
#         # Register hooks
#         self.base.register_forward_hook(self._forward_hook)
#         self.base.register_full_backward_hook(self._backward_hook)
        
#     def _forward_hook(self, module, inputs, output):
#         """Save input for K-FAC update"""
#         self._saved_input = inputs[0].detach()
        
#     def _backward_hook(self, module, grad_input, grad_output):
#         """Save gradient for K-FAC update"""
#         self._saved_grad_out = grad_output[0].detach()
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass with LoRA adaptation"""
#         base_out = self.base(x)
        
#         # LoRA path
#         x_d = self.dropout(x)
#         lora_mid = F.linear(x_d, self.lora_A)  # (batch, r)
#         lora_out = F.linear(lora_mid, self.lora_B)  # (batch, out)
        
#         return base_out + lora_out * self.scaling
    
#     def update_kfac_and_compute_preconditioned_grads(self, cfg, debug=False):
#         """
#         Update K-FAC statistics and precondition gradients.
#         Includes a `debug` flag to control verbose printing.
#         """
#         if debug:
#             print(f"\n{'='*20} GRIT DEBUG TRACE {'='*20}")
#             # Optional: Add a layer identifier if you have one
#             if hasattr(self.base, 'name_for_debug'):
#                 print(f"Layer: {self.base.name_for_debug}")

#         # === Step 1: Check Inputs for K-FAC Update ===
#         if self._saved_input is None or self._saved_grad_out is None:
#             if debug:
#                 print(f"[Step 1] K-FAC Inputs")
#                 print("  - [SKIPPING] _saved_input or _saved_grad_out is None.")
#                 print(f"{'='*53}\n")
#             return
        
#         if debug:
#             print(f"[Step 1] K-FAC Inputs")
#             print(f"  - Input tensor norm: {self._saved_input.norm().item():.6f}")
#             print(f"  - Grad-output tensor norm: {self._saved_grad_out.norm().item():.6f}")

#         X = self._saved_input
#         G_out = self._saved_grad_out
        
#         # === Step 2: K-FAC Statistics Update ===
#         if debug:
#             print(f"\n[Step 2] K-FAC Matrix Update")
#             norm_A_before = self.kfac.A.norm().item()
#             norm_G_before = self.kfac.G.norm().item()
        
#         self.kfac.update(X, G_out, debug=False) # The update always happens
        
#         if debug:
#             norm_A_after = self.kfac.A.norm().item()
#             norm_G_after = self.kfac.G.norm().item()
#             print(f"  - K-FAC A norm | Before: {norm_A_before:.6f} -> After: {norm_A_after:.6f}")
#             print(f"  - K-FAC G norm | Before: {norm_G_before:.6f} -> After: {norm_G_after:.6f}")
#             if abs(norm_A_after - norm_A_before) < 1e-9:
#                 print("  - [WARNING] K-FAC A matrix norm did not change. Statistics may not be accumulating.")
        
#         # === Step 3: Read Original Gradients ===
#         if self.lora_A.grad is None or self.lora_B.grad is None:
#             if debug:
#                 print(f"\n[Step 3] Original Gradients from Autograd")
#                 print("  - [SKIPPING] Gradients are None.")
#                 print(f"{'='*53}\n")
#             self._saved_input = None
#             self._saved_grad_out = None
#             return
        
#         grad_A_auto = self.lora_A.grad.clone().to(torch.float32)
#         grad_B_auto = self.lora_B.grad.clone().to(torch.float32)

#         if debug:
#             print(f"\n[Step 3] Original Gradients from Autograd")
#             norm_grad_A_orig = grad_A_auto.norm().item()
#             norm_grad_B_orig = grad_B_auto.norm().item()
#             print(f"  - Original grad norm A: {norm_grad_A_orig:.6f}")
#             print(f"  - Original grad norm B: {norm_grad_B_orig:.6f}")

#         # === Step 4: Damping and Inversion ===
#         A_inv = damp_and_invert(self.kfac.A.to(torch.float32), cfg.kfac_damping)
#         G_inv = damp_and_invert(self.kfac.G.to(torch.float32), cfg.kfac_damping)

#         if debug:
#             print(f"\n[Step 4] Damping and Inversion")
#             print(f"  - Damping factor: {cfg.kfac_damping}")
#             eye_G = torch.eye(G_inv.shape[0], device=G_inv.device)
#             print(f"  - Inverted G norm: {G_inv.norm().item():.4f} (Identity norm would be: {eye_G.norm().item():.4f})")
#             dist_G_from_eye = (G_inv - eye_G).norm().item()
#             print(f"  - Distance of G_inv from Identity: {dist_G_from_eye:.6f}")
#             if dist_G_from_eye < 1e-4:
#                 print("  - [WARNING] G_inv is nearly an identity matrix. The inversion may be ineffective.")

#         # === Step 5: Preconditioning (Multiplication) ===
#         grad_B_pre = G_inv @ grad_B_auto
#         grad_A_pre = grad_A_auto @ A_inv
        
#         if debug:
#             print(f"\n[Step 5] Preconditioning via Matrix Multiplication")
#             norm_grad_A_pre = grad_A_pre.norm().item()
#             norm_grad_B_pre = grad_B_pre.norm().item()
#             # We need the original norm from Step 3 for comparison
#             norm_grad_B_orig_for_check = grad_B_auto.norm().item()
#             print(f"  - Preconditioned grad norm A: {norm_grad_A_pre:.6f}")
#             print(f"  - Preconditioned grad norm B: {norm_grad_B_pre:.6f}")
#             if abs(norm_grad_B_orig_for_check - norm_grad_B_pre) < 1e-9:
#                 print("  - [CRITICAL] Norm of grad B was unchanged by preconditioning!")

#         grad_A_final = grad_A_pre
#         grad_B_final = grad_B_pre

#         # === Step 6 (Optional): Neural Reprojection ===
#         if cfg.reprojection_rank > 0:
#             if debug:
#                 print(f"\n[Step 6] Reprojection (Optional)")
            
#             self.kfac.compute_eigendecomp()
#             k = min(cfg.reprojection_rank, self.kfac.G_eigvecs.shape[1], self.kfac.A_eigvecs.shape[1])
            
#             if debug:
#                 print(f"  - Reprojecting onto rank k={k}")
            
#             if k > 0:
#                 top_g = self.kfac.G_eigvecs[:, -k:]
#                 top_a = self.kfac.A_eigvecs[:, -k:]
                
#                 grad_B_final = top_g @ (top_g.T @ grad_B_pre)
#                 grad_A_final = (grad_A_pre @ top_a) @ top_a.T
                
#                 if debug:
#                     norm_grad_A_final = grad_A_final.norm().item()
#                     norm_grad_B_final = grad_B_final.norm().item()
#                     print(f"  - Final grad norm A after reprojection: {norm_grad_A_final:.6f}")
#                     print(f"  - Final grad norm B after reprojection: {norm_grad_B_final:.6f}")

#         # === Step 7: Final Update ===
#         with torch.no_grad():
#             self.lora_A.grad.copy_(grad_A_final.to(self.lora_A.dtype))
#             self.lora_B.grad.copy_(grad_B_final.to(self.lora_B.dtype))

#         if debug:
#             print(f"\n[Step 7] Copied final gradients back to .grad attribute.")

#         # === Cleanup ===
#         self._saved_input = None
#         self._saved_grad_out = None

#         if debug:
#             print(f"{'='*22} END TRACE {'='*23}\n")    
    
#     def merge_to_base(self):
#         """Merge LoRA weights into base model"""
#         with torch.no_grad():
#             delta_W = (self.lora_B @ self.lora_A) * self.scaling
#             self.base.weight.data += delta_W
    
#     def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
#         """Get state dict for adapter parameters"""
#         return {
#             'lora_A': self.lora_A.data,
#             'lora_B': self.lora_B.data
#         }
    
#     def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]):
#         """Load adapter parameters from state dict"""
#         self.lora_A.data = state_dict['lora_A']
#         self.lora_B.data = state_dict['lora_B']






    # def update_kfac_and_compute_preconditioned_grads(self, cfg):
    #     """Update K-FAC statistics and compute preconditioned gradients"""
    #     if self._saved_input is None or self._saved_grad_out is None:
    #         return
        
    #     X = self._saved_input  # (batch, in)
    #     G_out = self._saved_grad_out  # (batch, out)
        
    #     # Update K-FAC statistics
    #     self.kfac.update(X, G_out)
        
    #     # Compute weight gradient
    #     grad_W = G_out.transpose(0, 1) @ X  # (out, in)
    #     grad_W32 = grad_W.to(torch.float32)
        
    #     # Compute K-FAC preconditioned gradient
    #     A_inv = damp_and_invert(self.kfac.A.to(torch.float32), cfg.kfac_damping)
    #     G_inv = damp_and_invert(self.kfac.G.to(torch.float32), cfg.kfac_damping)
    #     grad_W_pre = G_inv @ grad_W32 @ A_inv
        
    #     # Optional reprojection onto top eigenvectors
    #     if cfg.reprojection_rank > 0 and hasattr(self, 'reprojection_matrix'):
    #         # Compute eigendecomposition periodically
    #         self.kfac.compute_eigendecomp()
            
    #         # Get top-k eigenvectors
    #         k = min(cfg.reprojection_rank, self.kfac.G_eigvecs.shape[1], self.kfac.A_eigvecs.shape[1])
    #         top_g = self.kfac.G_eigvecs[:, -k:]  # (out, k)
    #         top_a = self.kfac.A_eigvecs[:, -k:]  # (in, k)
            
    #         # Project gradient onto top eigenvector subspace
    #         core = top_g.T @ grad_W_pre @ top_a  # (k, k)
    #         grad_W_pre = top_g @ core @ top_a.T
        
    #     # Map to LoRA parameter gradients
    #     with torch.no_grad():
    #         grad_A = (self.lora_B.T.to(torch.float32) @ grad_W_pre).to(self.lora_A.dtype)
    #         grad_B = (grad_W_pre @ self.lora_A.T.to(torch.float32)).to(self.lora_B.dtype)
            
    #         # Apply scaling
    #         grad_A *= self.scaling
    #         grad_B *= self.scaling
            
    #         # Set gradients
    #         self.lora_A.grad = grad_A
    #         self.lora_B.grad = grad_B
        
    #     # Clear saved tensors
    #     self._saved_input = None
    #     self._saved_grad_out = None
    
    
    
        # def update_kfac_and_compute_preconditioned_grads(self, cfg):
    #     """
    #     Update K-FAC statistics and precondition the existing gradients from autograd.
    #     This version correctly uses the gradients computed by loss.backward().
    #     """
    #     # === Step 1: Update K-FAC Statistics (No Change Here) ===
    #     # We still need the local input (X) and output-gradient (G_out) to
    #     # update our running averages for the covariance matrices.
    #     if self._saved_input is None or self._saved_grad_out is None:
    #         return
        
    #     print(f"[DEBUG] Input norm: {self._saved_input.norm().item():.4f}, Grad-output norm: {self._saved_grad_out.norm().item():.4f}")

    #     X = self._saved_input
    #     G_out = self._saved_grad_out
    #     self.kfac.update(X, G_out)

    #     # === Step 2: READ the True Gradients from Autograd ===
    #     # This is the most critical change. We check if autograd has computed
    #     # gradients for our LoRA parameters. If not, we do nothing.
    #     # Store norms before update
    #     norm_A_before = self.kfac.A.norm().item()
    #     norm_G_before = self.kfac.G.norm().item()
    #     print(f"[DEBUG] K-FAC A norm BEFORE update: {norm_A_before:.4f}, AFTER update: {self.kfac.A.norm().item():.4f}")
    #     print(f"[DEBUG] K-FAC G norm BEFORE update: {norm_G_before:.4f}, AFTER update: {self.kfac.G.norm().item():.4f}")
        
    #     if self.lora_A.grad is None or self.lora_B.grad is None:
    #         # Clear saved tensors for the next iteration and exit
    #         self._saved_input = None
    #         self._saved_grad_out = None
    #         return
        

    
    #     # We use .clone() to work with a copy, preventing any unintended
    #     # side-effects on the original gradient tensors.
    #     grad_A_auto = self.lora_A.grad.clone().to(torch.float32)
    #     grad_B_auto = self.lora_B.grad.clone().to(torch.float32)

    #     # === Step 3: Precondition the True Gradients ===
    #     # Now, we apply our K-FAC inverse matrices to transform these true
    #     # gradients into a more optimal update direction.
    #     A_inv = damp_and_invert(self.kfac.A.to(torch.float32), cfg.kfac_damping)
    #     G_inv = damp_and_invert(self.kfac.G.to(torch.float32), cfg.kfac_damping)

    #     # Precondition grad_B: Its geometry is defined by the output space (G).
    #     # Shape: [out, r] = [out, out] @ [out, r]
    #     grad_B_pre = G_inv @ grad_B_auto

    #     # Precondition grad_A: Its geometry is defined by the input space (A).
    #     # Shape: [r, in] = [r, in] @ [in, in]
    #     grad_A_pre = grad_A_auto @ A_inv

    #     # At this point, grad_A_pre and grad_B_pre are our final gradients
    #     # unless we are applying reprojection.
    #     grad_A_final = grad_A_pre
    #     grad_B_final = grad_B_pre

    #     # === Step 4 (Optional): Neural Reprojection ===
    #     # We project the preconditioned gradients onto the most significant
    #     # eigen-directions of the curvature to filter out noise.
    #     # Note: The original unreachable code bug is also fixed here.
    #     if cfg.reprojection_rank > 0:
    #         # We need to compute the eigendecomposition to find the subspaces
    #         self.kfac.compute_eigendecomp()
    #         k = min(cfg.reprojection_rank, self.kfac.G_eigvecs.shape[1], self.kfac.A_eigvecs.shape[1])
            
    #         if k > 0:
    #             # Get the top-k eigenvectors for both input and output spaces
    #             top_g = self.kfac.G_eigvecs[:, -k:]  # Shape: [out, k]
    #             top_a = self.kfac.A_eigvecs[:, -k:]  # Shape: [in, k]

    #             # Project grad_B_pre onto the subspace spanned by top_g
    #             # P_g = top_g @ top_g.T is the projection matrix
    #             grad_B_final = top_g @ (top_g.T @ grad_B_pre)

    #             # Project grad_A_pre onto the subspace spanned by top_a
    #             # P_a = top_a @ top_a.T is the projection matrix
    #             grad_A_final = (grad_A_pre @ top_a) @ top_a.T

    #     # === Step 5: UPDATE the Gradients In-Place ===
    #     # Finally, we overwrite the original .grad attributes with our new,
    #     # smarter, preconditioned (and possibly reprojected) gradients.
    #     # The optimizer will pick these up at the optimizer.step() call.
    #     with torch.no_grad():
    #         self.lora_A.grad.copy_(grad_A_final.to(self.lora_A.dtype))
    #         self.lora_B.grad.copy_(grad_B_final.to(self.lora_B.dtype))

    #     # === Step 6: Cleanup ===
    #     # Clear the saved tensors to be ready for the next forward/backward pass
    #     self._saved_input = None
    #     self._saved_grad_out = None