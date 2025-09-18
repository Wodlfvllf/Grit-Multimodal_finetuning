
# ============================================================================
# VQA Dataset
# ============================================================================
from typing import List, Dict
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from ..config import GRITConfig
from datasets import load_from_disk
from torchvision import transforms


system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
                    Your task is to analyze the provided chart image and respond to queries with concise answers, 
                    usually a single word, number, or short phrase.The charts include a variety of types (e.g., 
                    line charts, bar charts) and contain colors, labels, and text.Focus on delivering accurate, 
                    succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def format_data(sample):
    return {
      "images": [sample["image"]],
      "messages": [

          {
              "role": "system",
              "content": [
                  {
                      "type": "text",
                      "text": system_message
                  }
              ],
          },
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "image": sample["image"],
                  },
                  {
                      "type": "text",
                      "text": sample['query'],
                  }
              ],
          },
          {
              "role": "assistant",
              "content": [
                  {
                      "type": "text",
                      "text": sample["label"][0]
                  }
              ],
          },
      ]
      }
    
    
class VQADataset(Dataset):
    """VQA Dataset class for Qwen2-VL fine-tuning"""
    
    def __init__(self, 
                 processor: AutoProcessor, 
                 config: GRITConfig,
                 split: str = 'train'):
        
        self.vqa_data = load_from_disk("/root/GritProject/data/ChartQA")[split]
        self.processor = processor
        self.config = config
        self.pil_transform = transforms.Compose([
            transforms.Resize(32),  
            transforms.CenterCrop(32),
            transforms.ToTensor(), 
        ])
        
    def __len__(self):
        return len(self.vqa_data)
    
    def __getitem__(self, idx):
        sample = self.vqa_data[idx]
        formatted_sample = format_data(sample)
        
        # apply PIL resizing / center crop BEFORE processor
        pil_img = formatted_sample["images"][0]
        if isinstance(pil_img, str):
            pil_img = Image.open(pil_img).convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")
        resized_img_tensor = self.pil_transform(pil_img)  # [C,H,W]
        
        # Process the conversation using the processor
        text = self.processor.apply_chat_template(
            formatted_sample["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Process images and text together
        inputs = self.processor(
            text=[text],
            images=[ transforms.ToPILImage()(resized_img_tensor) ],
            padding=True,
            return_tensors="pt"
        )
        
        # Extract the processed data
        processed_item = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0) if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'].squeeze(0) if 'image_grid_thw' in inputs else None,
            'labels': inputs['input_ids'].squeeze(0).clone(),  # For causal LM, labels = input_ids
        }

        return processed_item
