import torch
import math
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Seq2SeqTrainingArguments,
    Qwen2VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from .data import VQADataset
from .config import GRITConfig
from .grit.manager import GRITManager
from .training.trainer import GritTrainer, GritCallback
from .utils.util import logger

def main():
    """Main training function"""
    
    # Create configuration
    config = GRITConfig()
    logger.info(f"Configuration: {config}")

    # Load base model and processor
    logger.info("Loading base model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.precision == "bf16" else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.config.use_cache = False

    # Create datasets
    train_dataset = VQADataset(processor, config, split='train')
    val_dataset = VQADataset(processor, config, split='val')

    eff_batch = config.batch_size * config.gradient_accumulation_steps
    total_steps = math.ceil(len(train_dataset) / eff_batch) * config.num_epochs
    logger.info(f"Total training steps: {total_steps}")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        bf16=config.precision == "bf16",
        max_grad_norm=1.5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=config.pin_memory,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing GRITManager on device: {device}")
    grit_manager = GRITManager(model, config, device)
    
    trainer = GritTrainer(
        grit_manager=grit_manager,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[GritCallback(grit_manager)],
    )
    
    # Train
    trainer.train()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()