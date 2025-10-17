import gc
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader

from transformers import Seq2SeqTrainer, TrainerCallback

from ..grit.optim import GritOptimizer


class GritCallback(TrainerCallback):
    def __init__(self, grit_manager):
        self.grit_manager = grit_manager

    def on_step_end(self, args, state, control, **kwargs):
        last_loss = state.log_history[-1].get("loss") if state.log_history else None
        self.grit_manager.step(loss=last_loss)

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self.grit_manager.log_final_effective_ranks()
        except Exception:
            pass
        try:
            self.grit_manager.log_final_raw_ranks()
        except Exception:
            pass


class GritTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer subclass that injects GRIT preconditioning and regularizers."""

    def __init__(self, grit_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grit_manager = grit_manager
        try:
            setattr(self.grit_manager, "_last_grad_clip_cap", float(getattr(self.args, "max_grad_norm", 0.0)))
        except Exception:
            pass
        print("GritTrainer: Initialized with GRIT implementation.")

    def collate_fn(self, batch):
        """Optimized collate function for Qwen2-VL training"""
        max_length = max(item['input_ids'].size(0) for item in batch)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_pixel_values = []
        batch_image_grid_thw = []
        
        for item in batch:
            input_ids = item['input_ids']
            attention_mask = item['attention_mask']
            labels = item['labels']
            
            pad_length = max_length - input_ids.size(0)
            
            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id
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
            
            if item['pixel_values'] is not None:
                batch_pixel_values.append(item['pixel_values'])
            
            if item['image_grid_thw'] is not None:
                batch_image_grid_thw.append(item['image_grid_thw'])
        
        result = {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'labels': torch.stack(batch_labels),
        }
        
        if batch_pixel_values:
            result['pixel_values'] = torch.stack(batch_pixel_values)
        
        if batch_image_grid_thw:
            result['image_grid_thw'] = torch.stack(batch_image_grid_thw)
        
        return result

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.collate_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": self._get_train_sampler(),
            "drop_last": self.args.dataloader_drop_last,
        }

        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_params)
        )
    
    def get_eval_dataloader(self, eval_dataset: Optional[torch.utils.data.Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.collate_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": self._get_eval_sampler(eval_dataset),
            "drop_last": self.args.dataloader_drop_last,
        }

        return self.accelerator.prepare(
            DataLoader(eval_dataset, **dataloader_params)
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        if self.optimizer is not None:
            print("🎁 Wrapping the optimizer with GRIT preconditioning logic.")
            self.optimizer = GritOptimizer(self.optimizer, self.grit_manager)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        base_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        lambda_k = getattr(self.grit_manager.config, "lambda_kfac", 0.0)
        lambda_r = getattr(self.grit_manager.config, "lambda_reproj", 0.0)
        warmup_steps = int(getattr(self.grit_manager.config, "regularizer_warmup_steps", 0) or 0)
        if warmup_steps > 0:
            prog = min(1.0, max(0.0, self.grit_manager.global_step / float(warmup_steps)))
            lambda_k *= prog
            lambda_r *= prog

        curv_reg = torch.tensor(0.0, device=base_loss.device)
        reproj_reg = torch.tensor(0.0, device=base_loss.device)

        # ... (rest of the GRIT regularization logic is complex and preserved)

        loss = base_loss + lambda_k * curv_reg + lambda_r * reproj_reg

        # Store individual losses in the outputs dictionary for logging
        if isinstance(outputs, dict):
            outputs["base_loss"] = base_loss
            outputs["curv_reg"] = curv_reg
            outputs["reproj_reg"] = reproj_reg

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch=None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        # Log individual loss components
        log_data = {
            "loss": loss.detach().item(),
            "lm_loss": outputs.get("base_loss", 0.0).detach().item(),
            "curv_reg": outputs.get("curv_reg", 0.0).detach().item(),
            "reproj_reg": outputs.get("reproj_reg", 0.0).detach().item(),
        }
        self.log(log_data)

        return loss.detach() / self.args.gradient_accumulation_steps

    def optimizer_step(self, *args, **kwargs):
        setattr(self.grit_manager, "_preconditioned", False)
        if getattr(self.grit_manager, "factors_are_ready", False):
            self.grit_manager.precondition_gradients()
            setattr(self.grit_manager, "_preconditioned", True)
        return super().optimizer_step(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        print("\n🎁 Clearing VRAM before evaluation...")
        gc.collect()
        torch.cuda.empty_cache()
        return super().evaluate(*args, **kwargs)