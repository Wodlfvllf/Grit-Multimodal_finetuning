

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
        
    def collate_fn(self, batch):
        """Fixed collate function that handles all fields properly"""
        max_length = max(item['input_ids'].size(0) for item in batch)
        
        batch_data = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'answer_labels': [],
            'pixel_values': [],
            'image_grid_thw': [],  # Added this
            'questions': [],
            'answers': [],
            'image_ids': []
        }
        
        for item in batch:
            # Pad sequences
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            pad_length = max_length - input_ids.size(0)
            if pad_length > 0:
                input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=attention_mask.dtype)])
                labels = torch.cat([labels, torch.full((pad_length,), -100, dtype=labels.dtype)])
            
            batch_data['input_ids'].append(input_ids)
            batch_data['attention_mask'].append(attention_mask)
            batch_data['labels'].append(labels)
            batch_data['answer_labels'].append(item['answer_label'])
            
            # Handle visual inputs
            if item['pixel_values'] is not None:
                batch_data['pixel_values'].append(item['pixel_values'])
            else:
                # If no pixel values, create a dummy tensor or handle appropriately
                batch_data['pixel_values'].append(torch.zeros(3, 224, 224))  # Dummy tensor
            
            # Handle image_grid_thw
            if item['image_grid_thw'] is not None:
                batch_data['image_grid_thw'].append(item['image_grid_thw'][0])
            else:
                # Default tensor if missing
                batch_data['image_grid_thw'].append(torch.tensor([1, 30, 40]))
            
            batch_data['questions'].append(item['question'])
            batch_data['answers'].append(item['answer'])
            batch_data['image_ids'].append(item['image_id'])
        
        # Stack tensors - NOW INCLUDING THE MISSING ONES
        result = {
            'input_ids': torch.stack(batch_data['input_ids']),
            'attention_mask': torch.stack(batch_data['attention_mask']),
            'labels': torch.stack(batch_data['labels']),
            'answer_labels': torch.stack(batch_data['answer_labels']),
            'pixel_values': torch.stack(batch_data['pixel_values']),  # ADDED THIS
            'image_grid_thw': torch.stack(batch_data['image_grid_thw']),  # ADDED THIS
            'questions': batch_data['questions'],
            'answers': batch_data['answers'],
            'image_ids': batch_data['image_ids']
        }
        
        return result
    
    def _create_optimizer(self):
        """Create optimizer for GRIT parameters"""
        grit_params = []
        for wrapper in self.model.grit_wrappers:
            grit_params.extend([wrapper.lora_A, wrapper.lora_B])
        
        all_params = grit_params + list(self.classification_head.parameters())
        
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
        lm_loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            batch['labels'].view(-1),
            ignore_index=-100
        )
        
        # Get hidden states for classification
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
            pooled_hidden = last_hidden.mean(dim=1)
        else:
            # Fallback
            pooled_hidden = outputs.logits.mean(dim=1)[:, :self.model.base_model.config.hidden_size]
        
        # Classification loss
        classification_logits = self.classification_head(pooled_hidden)
        valid_mask = batch['answer_labels'] != -100
        
        if valid_mask.sum() > 0:
            cls_loss = F.cross_entropy(
                classification_logits[valid_mask],
                batch['answer_labels'][valid_mask]
            )
        else:
            cls_loss = torch.tensor(0.0, device=classification_logits.device)
        
        # Curvature regularization
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
        total_loss = 0.7 * lm_loss + 0.3 * cls_loss + curvature_reg + reprojection_reg
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'cls_loss': cls_loss,
            'curvature_reg': curvature_reg,
            'reprojection_reg': reprojection_reg,
            'classification_logits': classification_logits
        }
        
    def compute_accuracy(self, logits, labels):
        """Compute accuracy"""
        valid_mask = labels != -100
        if valid_mask.sum() == 0:
            return 0.0
        
        predictions = torch.argmax(logits[valid_mask], dim=-1)
        correct = (predictions == labels[valid_mask]).float()
        return correct.mean().item()