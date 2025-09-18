
# ============================================================================
# Main Training Script
# ============================================================================
from .utils import KFACStatistics, damp_and_invert
from .data import VQADataset
from .models import GRITModel, replace_linear_with_grit, LinearWithGRIT
from .config import GRITConfig
from .training import GRITTrainer
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
    
    # Dataset paths
    DATASET_ROOT = '/root/GritProject/data/dataset/'
    ANSWER_SPACE_PATH = '/root/GritProject/data/dataset/answer_space.txt'
    print(ANSWER_SPACE_PATH)
    csv_path = '/root/GritProject/data/dataset/data.csv'
    image_root = '/root/GritProject/data/dataset/images/'
    
    # Load answer space
    with open(ANSWER_SPACE_PATH, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    classes_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_classes = {idx: cls for idx, cls in enumerate(classes)}
    
    logger.info(f"Loaded {len(classes)} answer classes")
    
    # Load data
    df = pd.read_csv(csv_path)
    all_data = [
        {'question': q, 'answer': a.split(',')[0], 'image_id': i}
        for q, a, i in zip(df['question'], df['answer'], df['image_id'])
    ]
    
    # Split data
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    logger.info(f"Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
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
    train_dataset = VQADataset(train_data, classes_to_idx, processor, image_root, config)
    val_dataset = VQADataset(val_data, classes_to_idx, processor, image_root, config)
    
    # Create trainer
    trainer = GRITTrainer(
        model, train_dataset, val_dataset, config,
        classes_to_idx, idx_to_classes
    )
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params += sum(p.numel() for p in trainer.classification_head.parameters())
    
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