

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