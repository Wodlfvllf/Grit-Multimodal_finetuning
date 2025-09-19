

class LoRATrainer:
    """Trainer for standard LoRA fine-tuning"""
    
    def __init__(self, 
                 model,  # PEFT model
                 processor,
                 train_dataset: VQADataset,
                 val_dataset: Optional[VQADataset], 
                 config: LoRAConfig
                 ):
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.training_mode = "LoRA"
        
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
        
        # Optimizer - only optimize trainable parameters
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def collate_fn(self, batch):
        """Collate function for LoRA training (same as GRIT)"""
        # Determine max length for padding
        max_length = max(item['input_ids'].size(0) for item in batch)
        
        # Initialize batch containers
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        
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
                    torch.full((pad_length,), -100, dtype=labels.dtype)
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
        """Create optimizer for LoRA parameters only"""
        # Get only trainable parameters (LoRA adapters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        total_steps = len(self.train_loader) * self.config.num_epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6
        )
        
    def train_epoch(self, epoch):
        """Training epoch for standard LoRA"""
        self.model.train()
        epoch_loss = 0.0
        num_update_steps = 0
        
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        
        for batch_idx, batch in progress_bar:
            # Forward pass
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch.get('pixel_values'),
                image_grid_thw=batch.get('image_grid_thw'),
                labels=batch['labels']
            )
            
            # Simple cross-entropy loss (no GRIT regularization)
            loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every N steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update metrics
                actual_loss = loss.item() * self.config.gradient_accumulation_steps
                epoch_loss += actual_loss
                num_update_steps += 1
                
                # Log
                with open("/root/GritProject/lora_log.txt", "a") as f:
                    f.write(f"Epoch {epoch+1}, Update Step {num_update_steps}, Loss: {actual_loss}\n")
                    f.flush()
                
                # Scheduler step
                self.scheduler.step()
            
            # Progress bar
            true_loss = loss.item() * self.config.gradient_accumulation_steps
            progress_bar.set_postfix({'loss': f"{true_loss:.4f}"})
        
        # Average loss
        avg_loss = epoch_loss / max(num_update_steps, 1)
        self.train_losses.append(avg_loss)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        return avg_loss