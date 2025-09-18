
from typing import List, Dict
import torch
import torch.nn as nn
from .grit_layer import LinearWithGRIT
from .replace_grit_modules import replace_linear_with_grit
import logging
from ..config import GRITConfig
logger = logging.getLogger(__name__)

class GRITModel(nn.Module):
    """Main GRIT model wrapper"""
    
    def __init__(self, base_model: nn.Module, config: GRITConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Store GRIT wrappers
        self.grit_wrappers: List[LinearWithGRIT] = []
        self._name_to_wrapper: Dict[str, LinearWithGRIT] = {}
        
        # Apply GRIT to target modules
        self._apply_grit_to_model()
        
        # Freeze base parameters
        for p in self.base_model.parameters():
            p.requires_grad = False
        
        # Enable gradients for adapter parameters
        for wrapper in self.grit_wrappers:
            wrapper.lora_A.requires_grad = True
            wrapper.lora_B.requires_grad = True
    
    def _apply_grit_to_model(self):
        """Replace target modules with GRIT wrappers"""
        target_modules = self.config.target_modules
        all_names = [name for name, _ in self.base_model.named_modules()]
        
        for name in all_names:
            if any(target in name for target in target_modules):
                try:
                    # Check if it's a Linear module
                    parts = name.split(".")
                    parent = self.base_model
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    module = getattr(parent, parts[-1])
                    
                    if isinstance(module, nn.Linear):
                        wrapper = replace_linear_with_grit(self.base_model, name, self.config)
                        self.grit_wrappers.append(wrapper)
                        self._name_to_wrapper[name] = wrapper
                        logger.info(f"Replaced {name} with GRIT wrapper")
                except Exception as e:
                    logger.warning(f"Failed to replace {name}: {e}")
    
    def forward(self, *args, **kwargs):
        """Forward pass through base model"""
        return self.base_model(*args, **kwargs)
    
    def update_grit_gradients(self):
        """Update gradients using K-FAC preconditioning"""
        for wrapper in self.grit_wrappers:
            wrapper.update_kfac_and_compute_preconditioned_grads(self.config)
    
    def merge_all_adapters(self):
        """Merge all adapters into base model"""
        for wrapper in self.grit_wrappers:
            wrapper.merge_to_base()
    
    def save_adapters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Save adapter state"""
        state = {}
        for name, wrapper in self._name_to_wrapper.items():
            state[name] = wrapper.get_adapter_state_dict()
        return state
    
    def load_adapters(self, state: Dict[str, Dict[str, torch.Tensor]]):
        """Load adapter state"""
        for name, adapter_state in state.items():
            if name in self._name_to_wrapper:
                self._name_to_wrapper[name].load_adapter_state_dict(adapter_state)
