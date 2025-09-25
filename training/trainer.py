

# ============================================================================
# GRIT Trainer
# ============================================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from ..config import GRITConfig
from ..data import VQADataset
from ..models import GRITModel, LinearWithGRIT

class GRITTrainer:
    """Trainer for GRIT fine-tuning"""
    
    def __init__(self, 
                 model: GRITModel, 
                 processor,
                 train_dataset: VQADataset,
                 val_dataset: Optional[VQADataset], 
                 config: GRITConfig
                 ):
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True, 
            num_workers=config.num_workers,
            pin_memory=True, 
            collate_fn=self.collate_fn
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True, 
                collate_fn=self.collate_fn
            )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def collate_fn(self, batch):
        """Optimized collate function for Qwen2-VL training"""
        
        # Determine max length for padding
        max_length = max(item['input_ids'].size(0) for item in batch)
        
        # Initialize batch containers
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        
        # Optional fields for debugging/evaluation
        batch_questions = []
        batch_answers = []
        batch_image_ids = []
        
        for item in batch:
            # Pad sequences to max_length
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            pad_length = max_length - input_ids.size(0)
            
            if pad_length > 0:
                pad_token_id = self.processor.tokenizer.pad_token_id
                input_ids = torch.cat([
                    input_ids, 
                    torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype)
                ])
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.zeros(pad_length, dtype=attention_mask.dtype)
                ])
                labels = torch.cat([
                    labels, 
                    torch.full((pad_length,), -100, dtype=labels.dtype)  # -100 for ignore_index
                ])
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            
            # Handle visual inputs
            if item['pixel_values'] is not None:
                batch_pixel_values.append(item['pixel_values'])
            
            if item['image_grid_thw'] is not None:
                batch_image_grid_thw.append(item['image_grid_thw'])
        
        # Create final batch dictionary
        result = {
            # Essential fields for Qwen2-VL forward pass
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'labels': torch.stack(batch_labels),
        }
        
        # Add visual inputs if present
        if batch_pixel_values:
            result['pixel_values'] = torch.stack(batch_pixel_values)
        
        if batch_image_grid_thw:
            result['image_grid_thw'] = torch.stack(batch_image_grid_thw)
        
        return result

    
    def _create_optimizer(self):
        """Create optimizer for GRIT parameters"""
        grit_params = []
        for wrapper in self.model.grit_wrappers:
            grit_params.extend([wrapper.lora_A, wrapper.lora_B])
        
        all_params = grit_params
        
        return torch.optim.AdamW(
            all_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config.num_epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6
        )
        
    def compute_grit_loss(self, outputs, batch):
        """Compute GRIT-specific loss with regularization"""
        # Language modeling loss
        lm_loss = outputs.loss

        curvature_reg = 0.0
        for wrapper in self.model.grit_wrappers:
            if wrapper.kfac.A is not None and wrapper.kfac.G is not None:
                A_trace = torch.trace(wrapper.kfac.A)
                G_trace = torch.trace(wrapper.kfac.G)
                curvature_reg += (A_trace + G_trace) / 2
        
        curvature_reg = self.config.lambda_curvature * curvature_reg / len(self.model.grit_wrappers)
        
        # Reprojection regularization (L2 norm of parameters)
        reprojection_reg = 0.0
        for wrapper in self.model.grit_wrappers:
            reprojection_reg += torch.norm(wrapper.lora_A) + torch.norm(wrapper.lora_B)
        
        reprojection_reg = self.config.lambda_reprojection * reprojection_reg / len(self.model.grit_wrappers)
        
        # Total loss
        total_loss = 0.7 * lm_loss + 0.15 * curvature_reg + 0.15 * reprojection_reg
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'curvature_reg': curvature_reg,
            'reprojection_reg': reprojection_reg,
        }
        
    def compute_accuracy(self, logits, labels):
        """Compute accuracy"""
        valid_mask = labels != -100
        if valid_mask.sum() == 0:
            return 0.0
        
        predictions = torch.argmax(logits[valid_mask], dim=-1)
        correct = (predictions == labels[valid_mask]).float()
        return correct.mean().item()
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        num_update_steps = 0  # Count gradient update steps, not mini-batches
        
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        
        #----------------------------------DEBUG--------------------------------------------#
        grit_wrappers = [m for m in self.model.modules() if isinstance(m, LinearWithGRIT)]
        if not grit_wrappers:
            raise ValueError("No LinearWithGRIT layers found to inspect.")
        layer_to_inspect = grit_wrappers[0] 
        
        # Store norms for the current accumulation cycle
        grad_norms_A = []
        grad_norms_B = []
        #----------------------------------DEBUG--------------------------------------------#
        
        for batch_idx, batch in progress_bar:
            # Forward pass
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch.get('pixel_values'),
                image_grid_thw=batch['image_grid_thw'],
                labels=batch['labels']
            )
            
            # Compute loss and scale it down for accumulation
            loss_dict = self.compute_grit_loss(outputs, batch)
            loss = loss_dict['total_loss'] / self.config.gradient_accumulation_steps
            
            # Backward pass (gradients accumulate automatically)
            loss.backward()
            
            # === Gradient Inspection Step ===
            # After backward(), but before the optimizer step.
            # Check if the gradient exists before trying to access its norm
            if layer_to_inspect.lora_A.grad is not None:
                norm_A = layer_to_inspect.lora_A.grad.norm().item()
                norm_B = layer_to_inspect.lora_B.grad.norm().item()
                grad_norms_A.append(norm_A)
                grad_norms_B.append(norm_B)
            # ================================
            
            # === LOG THE ACCUMULATED NORMS BEFORE PRECONDITIONING ===
            print(f"\n--- Update Step {num_update_steps + 1} (Batch {batch_idx + 1}) ---")
            print(f"Norms of lora_A.grad over {len(grad_norms_A)} accumulation steps: {grad_norms_A}")
            print(f"Norms of lora_B.grad over {len(grad_norms_B)} accumulation steps: {grad_norms_B}")
            print(f"Final accumulated norm for A: {layer_to_inspect.lora_A.grad.norm().item():.4f}")
            print(f"Final accumulated norm for B: {layer_to_inspect.lora_B.grad.norm().item():.4f}")
            # =========================================================

            
            # Only update weights every N steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Apply GRIT preconditioning
                self.model.update_grit_gradients()
                
                # === LOG THE NORMS AFTER PRECONDITIONING ===
                print(f"Norm of A.grad AFTER preconditioning: {layer_to_inspect.lora_A.grad.norm().item():.4f}")
                print(f"Norm of B.grad AFTER preconditioning: {layer_to_inspect.lora_B.grad.norm().item():.4f}")
                print("-------------------------------------------------")
                # ===========================================
            
                # Gradient clipping
                # all_params = [p for w in self.model.grit_wrappers for p in [w.lora_A, w.lora_B]]
                all_params = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update metrics (only after complete accumulation cycle)
                # Note: loss.item() is the scaled loss, so we multiply back to get true loss
                actual_loss = loss.item() * self.config.gradient_accumulation_steps
                epoch_loss += actual_loss
                num_update_steps += 1
                
                # Log the true loss (not scaled)
                with open("/root/GritProject/log.txt", "a") as f:
                    f.write(f"Epoch {epoch+1}, Update Step {num_update_steps}, Loss: {actual_loss}\n")
                    f.flush()
                
                # Scheduler step (should be per update, not per mini-batch)
                self.scheduler.step()
            
            # Progress bar shows the true loss (unscaled)
            true_loss = loss.item() * self.config.gradient_accumulation_steps
            progress_bar.set_postfix({
                'loss': f"{true_loss:.4f}",
                'lm_loss': f"{loss_dict['lm_loss'].item():.4f}",
            })
        
        # Average loss over update steps, not mini-batches
        avg_loss = epoch_loss / max(num_update_steps, 1)
        self.train_losses.append(avg_loss)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, epoch):
        """Validation loop"""
        if not self.val_dataset:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation"):
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch.get('pixel_values'),
                    image_grid_thw=batch.get('image_grid_thw'),
                    labels=batch['labels']
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1} - Val Loss: {avg_loss:.4f}")
        
        self.model.train()
        return avg_loss
    
    def train(self):
        """Main training loop"""
        logger.info("Starting GRIT training...")
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)
            val_acc = None  # Placeholder if accuracy is not computed
            # Save best model
            if val_acc is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(f"best_grit_model.pt")
                logger.info(f"Saved best model with val acc: {val_acc:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"grit_checkpoint_epoch_{epoch+1}.pt")
        
        logger.info("GRIT training completed!")
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint = {
            'epoch': len(self.train_losses),
            'model_adapters': self.model.save_adapters(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        save_path = Path("checkpoints") / filename
        save_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")