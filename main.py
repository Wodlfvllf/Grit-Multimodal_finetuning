
# ============================================================================
# Main Training Script
# ============================================================================
from .utils import KFACStatistics, damp_and_invert
from .data import VQADataset
from .models import GRITModel, replace_linear_with_grit, LinearWithGRIT
from .config import GRITConfig, LoRAConfig
from .training import GRITTrainer, LoRATrainer
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict
from .utils.util import *

def main():
    """Main training function"""
        
    # Create configuration
    config = GRITConfig()
    logger.info(f"Configuration: {config}")
    
    # Load base model and processor
    logger.info("Loading base model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.mixed_precision else torch.float32,
        device_map=config.device
    )
    
    processor = AutoProcessor.from_pretrained(config.model_name)
    
    # Create GRIT model
    logger.info("Creating GRIT model...")
    model = GRITModel(base_model, config)
    model = model.to(config.device)
    
    # Create datasets
    train_dataset = VQADataset(processor, config, split='train')
    val_dataset = VQADataset(processor, config, split='val')

    # Create trainer
    trainer = GRITTrainer(
        model, processor, train_dataset, val_dataset, config
    )
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Train
    results = trainer.train()
    
    # Save final model
    trainer.save_checkpoint("final_grit_model.pt")
    
    logger.info("Training completed successfully!")
    return results

if __name__ == "__main__":
    main()