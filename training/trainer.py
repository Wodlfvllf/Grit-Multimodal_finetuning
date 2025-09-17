

# ============================================================================
# GRIT Trainer
# ============================================================================

class GRITTrainer:
    """Trainer for GRIT fine-tuning"""
    
    def __init__(self, model: GRITModel, train_dataset: VQADataset,
                 val_dataset: Optional[VQADataset], config: GRITConfig,
                 classes_to_idx: Dict[str, int], idx_to_classes: Dict[int, str]):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.classes_to_idx = classes_to_idx
        self.idx_to_classes = idx_to_classes
        self.n_classes = len(classes_to_idx)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
            pin_memory=True, collate_fn=self.collate_fn
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size,
                shuffle=False, num_workers=config.num_workers,
                pin_memory=True, collate_fn=self.collate_fn
            )
        
        # Classification head for VQA
        self.classification_head = nn.Linear(
            self.model.base_model.config.hidden_size,
            self.n_classes
        ).to(config.device)
        
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