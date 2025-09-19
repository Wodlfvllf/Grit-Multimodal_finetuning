

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