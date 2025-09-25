# ============================================================================
# Main Training Script with PEFT LoRA Integration
# ============================================================================
from .utils import KFACStatistics, damp_and_invert
from .data import VQADataset
from .models import GRITModel, replace_linear_with_grit, LinearWithGRIT
from .config import LoRAConfig
from .training import LoRATrainer
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict
from .utils.util import *

def create_peft_config(config: LoRAConfig):
    """Create PEFT LoRA configuration"""
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # or TaskType.FEATURE_EXTRACTION for vision tasks
        inference_mode=False,
        r=config.rank,  # rank
        lora_alpha=config.alpha,  # scaling parameter
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        # Optional: specify modules to save (useful for vision-language models)
        modules_to_save=["embed_tokens", "lm_head"],  # commonly saved modules
    )
    return peft_config

def main():
    """Main training function with PEFT LoRA"""
        
    # Create configuration
    config = LoRAConfig()
    logger.info(f"Configuration: {config}")
    
    # Load base model and processor
    logger.info("Loading base model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        device_map=config.device,

    )
    
    processor = AutoProcessor.from_pretrained(config.model_name)
    
    # Create PEFT configuration
    logger.info("Creating PEFT LoRA configuration...")
    peft_config = create_peft_config(config)
    
    # Apply PEFT LoRA to the model
    logger.info("Applying PEFT LoRA to model...")
    model = get_peft_model(base_model, peft_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    # Move model to device
    model = model.to(config.device)
    
    # Create datasets
    train_dataset = VQADataset(processor, config, split='train')
    val_dataset = VQADataset(processor, config, split='val')

    # Create trainer
    trainer = LoRATrainer(
        model, processor, train_dataset, val_dataset, config
    )
    
    # Calculate parameters (alternative manual calculation)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Train
    results = trainer.train()
    
    # Save PEFT model (this saves only the LoRA adapters)
    logger.info("Saving PEFT LoRA adapters...")
    model.save_pretrained("./lora_adapters")
    processor.save_pretrained("./lora_adapters")
    
    # Optional: Save full model if needed
    # trainer.save_checkpoint("final_lora_model.pt")
    
    logger.info("Training completed successfully!")
    return results

if __name__ == "__main__":
    main()