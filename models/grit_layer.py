# ============================================================================
# GRIT Layer Implementation
# ============================================================================

class LinearWithGRIT(nn.Module):
    """Linear layer with GRIT adaptation"""
    
    def __init__(self, orig_linear: nn.Linear, cfg):
        super().__init__()
        self.base = orig_linear  # Keep original linear frozen
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        
        # GRIT/LoRA parameters
        self.r = cfg.rank
        self.alpha = cfg.alpha
        self.scaling = self.alpha / max(1, self.r)
        self.dropout_p = cfg.dropout
        
        # Initialize LoRA parameters
        device = next(orig_linear.parameters()).device
        dtype = next(orig_linear.parameters()).dtype
        
        self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=device, dtype=dtype))
        
        # Initialize using Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(self.dropout_p)
        
        # K-FAC statistics
        self.kfac = KFACStatistics(
            self.in_features, 
            self.out_features, 
            momentum=cfg.kfac_momentum, 
            device=device, 
            dtype=torch.float32
        )
        
        # Reprojection matrix (will be computed during training)
        self.reprojection_matrix = None
        self.reprojection_rank = cfg.reprojection_rank
        
        # Saved tensors for backward pass
        self._saved_input = None
        self._saved_grad_out = None
        
        # Register hooks
        self.base.register_forward_hook(self._forward_hook)
        self.base.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, inputs, output):
        """Save input for K-FAC update"""
        self._saved_input = inputs[0].detach()
        
    def _backward_hook(self, module, grad_input, grad_output):
        """Save gradient for K-FAC update"""
        self._saved_grad_out = grad_output[0].detach()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        base_out = self.base(x)
        
        # LoRA path
        x_d = self.dropout(x)
        lora_mid = F.linear(x_d, self.lora_A)  # (batch, r)
        lora_out = F.linear(lora_mid, self.lora_B)  # (batch, out)
        
        return base_out + lora_out * self.scaling
    
    def update_kfac_and_compute_preconditioned_grads(self, cfg):
        """Update K-FAC statistics and compute preconditioned gradients"""
        if self._saved_input is None or self._saved_grad_out is None:
            return
        
        X = self._saved_input  # (batch, in)
        G_out = self._saved_grad_out  # (batch, out)
        
        # Update K-FAC statistics
        self.kfac.update(X, G_out)
        
        # Compute weight gradient
        grad_W = G_out.transpose(0, 1) @ X  # (out, in)
        grad_W32 = grad_W.to(torch.float32)
        
        # Compute K-FAC preconditioned gradient
        A_inv = damp_and_invert(self.kfac.A.to(torch.float32), cfg.kfac_damping)
        G_inv = damp_and_invert(self.kfac.G.to(torch.float32), cfg.kfac_damping)
        grad_W_pre = G_inv @ grad_W32 @ A_inv
        
        # Optional reprojection onto top eigenvectors
        if cfg.reprojection_rank > 0 and hasattr(self, 'reprojection_matrix'):
            # Compute eigendecomposition periodically
            self.kfac.compute_eigendecomp()
            
            # Get top-k eigenvectors
            k = min(cfg.reprojection_rank, self.kfac.G_eigvecs.shape[1], self.kfac.A_eigvecs.shape[1])
            top_g = self.kfac.G_eigvecs[:, -k:]  # (out, k)
            top_a = self.kfac.A_eigvecs[:, -k:]  # (in, k)
            
            # Project gradient onto top eigenvector subspace
            core = top_g.T @ grad_W_pre @ top_a  # (k, k)
            grad_W_pre = top_g @ core @ top_a.T
        
        # Map to LoRA parameter gradients
        with torch.no_grad():
            grad_A = (self.lora_B.T.to(torch.float32) @ grad_W_pre).to(self.lora_A.dtype)
            grad_B = (grad_W_pre @ self.lora_A.T.to(torch.float32)).to(self.lora_B.dtype)
            
            # Apply scaling
            grad_A *= self.scaling
            grad_B *= self.scaling
            
            # Set gradients
            self.lora_A.grad = grad_A
            self.lora_B.grad = grad_B
        
        # Clear saved tensors
        self._saved_input = None
        self._saved_grad_out = None
    
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
