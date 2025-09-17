
# ============================================================================
# VQA Dataset
# ============================================================================

class VQADataset(Dataset):
    """VQA Dataset class"""
    
    def __init__(self, vqa_data: List[Dict], classes_to_idx: Dict[str, int],
                 processor: AutoProcessor, image_root_path: str, config: GRITConfig):
        self.vqa_data = vqa_data
        self.classes_to_idx = classes_to_idx
        self.processor = processor
        self.image_root_path = image_root_path
        self.config = config
    
    def __len__(self):
        return len(self.vqa_data)
    
    def __getitem__(self, idx):
        item = self.vqa_data[idx]
        
        # Load image
        image_path = os.path.join(self.image_root_path, f"{item['image_id']}.png")
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='white')
            print(f"Could not load image {image_path}")
        
        question = item['question']
        answer = item['answer']
        
        # Create conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]
        
        # Process with processor
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        
        inputs = self.processor(
            text=[text], images=[image],
            padding=True, truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        answer_idx = self.classes_to_idx.get(answer, -100)
        # print(inputs.get('image_grid_thw', torch.tensor([[1, 30, 40]])))
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze() if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs.get('image_grid_thw', torch.tensor([1, 30, 40])),
            'labels': inputs['input_ids'].squeeze(),
            'answer_label': torch.tensor(answer_idx, dtype=torch.long),
            'question': question,
            'answer': answer,
            'image_id': item['image_id']
        }
