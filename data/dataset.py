
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
    """VQA Dataset class"""
    
    def __init__(self, vqa_data: List[Dict], classes_to_idx: Dict[str, int],
                 processor: AutoProcessor, image_root_path: str, config: GRITConfig):
        self.vqa_data = load_from_disk("/root/GritProject/data/ChartQA")['train']
        self.dataset = [format_data(sample) for sample in self.vqa_data]
        self.processor = processor
        self.config = config
        self.system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
                                Your task is to analyze the provided chart image and respond to queries with concise answers, 
                                usually a single word, number, or short phrase.The charts include a variety of types (e.g., 
                                line charts, bar charts) and contain colors, labels, and text.Focus on delivering accurate, 
                                succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
    
    def __len__(self):
        return len(self.vqa_data)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text_input = self.processor.apply_chat_template(item['messages'], tokenize=False)
        image_input = self.processor(images=item['images'], return_tensors="pt")

        return {
            'input_ids': text_input['input_ids'].squeeze(),
            'attention_mask': text_input['attention_mask'].squeeze(),
            'pixel_values': image_input['pixel_values'].squeeze() if 'pixel_values' in image_input else None,
            'image_grid_thw': image_input.get('image_grid_thw', torch.tensor([1, 30, 40])),
            'labels': text_input['input_ids'].squeeze(),
            'answer_label': torch.tensor(item['answer_label'], dtype=torch.long),
            'question': item['question'],
            'answer': item['answer'],
            'image_id': item['image_id']
        }
        
# if __name__ == "__main__":
    
