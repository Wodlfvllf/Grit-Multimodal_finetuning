

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
                 processor,  # Store processor
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
        # print(outputs)
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
        total_loss = 0.7 * lm_loss + curvature_reg + reprojection_reg
        
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
        """Train for one epoch"""
        self.model.train()        
        epoch_loss = 0.0
        # epoch_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # logger.info(f"Epoch {epoch+1} - Entered training loop, batch {batch_idx+1}/{len(self.train_loader)}")
            # Move to device
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            # logger.info(f"Batch keys: {list(batch.keys())}")
            # logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
            # logger.info(f"Pixel values shape: {batch['pixel_values'].shape}")
            # logger.info(f"Image grid thw shape: {batch['image_grid_thw'].shape}")
            # print(batch['image_grid_thw'].shape)
            # Forward pass
            if self.config.mixed_precision:
                logger.info(f"Epoch {epoch+1} - Using mixed precision, batch {batch_idx+1}/{len(self.train_loader)}")
                with autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        pixel_values=batch.get('pixel_values'),
                        image_grid_thw = batch['image_grid_thw']
                        
                    )
                    logger.info(f"Model outputs obtained")
                    loss_dict = self.compute_grit_loss(outputs, batch)
                    loss = loss_dict['total_loss'] / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Update GRIT gradients with K-FAC
                    self.model.update_grit_gradients()
                    
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    all_params = [p for w in self.model.grit_wrappers for p in [w.lora_A, w.lora_B]]
                    # all_params.extend(list(self.classification_head.parameters()))
                    torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # logger.info(f"Epoch {epoch+1} - Using standard precision, batch {batch_idx+1}/{len(self.train_loader)}")
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pixel_values=batch.get('pixel_values'),
                    image_grid_thw = batch['image_grid_thw'],
                    labels=batch['labels']   # <-- add this!

                )
                loss_dict = self.compute_grit_loss(outputs, batch)
                loss = loss_dict['total_loss'] / self.config.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    with open("/root/GritProject/log.txt", "a") as f:
                        f.write(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}\n")
                        f.flush()
                    # Update GRIT gradients with K-FAC
                    self.model.update_grit_gradients()
                    
                    # Gradient clipping
                    all_params = [p for w in self.model.grit_wrappers for p in [w.lora_A, w.lora_B]]
                    # all_params.extend(list(self.classification_head.parameters()))
                    torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            # accuracy = self.compute_accuracy(loss_dict['classification_logits'], batch['answer_labels'])
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            # epoch_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config.gradient_accumulation_steps:.4f}",
                # 'acc': f"{accuracy:.4f}",
                'lm_loss': f"{loss_dict['lm_loss'].item():.4f}",
                # 'cls_loss': f"{loss_dict['cls_loss'].item():.4f}"
            })
            
            self.scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        # avg_accuracy = epoch_accuracy / num_batches
        
        self.train_losses.append(avg_loss)
        # self.train_accuracies.append(avg_accuracy)
        
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
    
    # def train(self):
    #     """Main training loop"""
    #     logger.info("Starting GRIT training...")
    #     best_val_acc = 0.0
        
    #     for epoch in range(self.config.num_epochs):
    #         # Train
    #         train_loss = self.train_epoch(epoch)

    #         # Validate
    #         val_loss = self.validate(epoch)
    #         val_acc = None  # Placeholder if accuracy is not computed
    #         # Save best model
    #         if val_acc is not None and val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             self.save_checkpoint(f"best_grit_model.pt")
    #             logger.info(f"Saved best model with val acc: {val_acc:.4f}")
            
    #         # Regular checkpoint
    #         if (epoch + 1) % 5 == 0:
    #             self.save_checkpoint(f"grit_checkpoint_epoch_{epoch+1}.pt")
        
    #     logger.info("GRIT training completed!")
    #     return {
    #         'train_losses': self.train_losses,
    #         'val_losses': self.val_losses,
    #     }
    def train(self):
        """Main training loop"""
        logger.info("Starting GRIT training...")
        best_val_acc = 0.0

        # open log file in append mode
        log_path = "log.txt"
        with open(log_path, "a") as f:
            for epoch in range(self.config.num_epochs):
                # Train
                train_loss = self.train_epoch(epoch)

                # Validate
                val_loss = self.validate(epoch)
                val_acc = None  # Placeholder if accuracy is not computed

                # Save losses to log.txt
                f.write(f"Epoch {epoch+1}/{self.config.num_epochs} "
                        f"- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
                f.flush()  # ensure it's written immediately

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
            # 'classification_head': self.classification_head.state_dict(),
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