# CHANGELOG.md

This file documents the architectural refactoring of the `grit_multimodal_finetuning` repository to align with the design patterns of the original GRIT implementation (`iclr2026-submission-18146`).

## Summary

The primary goal of this refactoring was to improve modularity, leverage a robust training framework, and adopt the core algorithmic components of the reference GRIT implementation while critically preserving the project's unique multimodal data handling capabilities.

## Changes

- **Created a Safe Backup**: A complete backup of the original `grit_multimodal_finetuning` directory was created at `grit_multimodal_finetuning_backup_20251016` before any modifications were made.

- **Adopted a Modular `grit` Subdirectory**: To mirror the structure of the reference repository, a new `grit_multimodal_finetuning/grit/` subdirectory was created. The core algorithmic components from the original implementation were copied here:
  - `grit/optim.py`: Contains the `GritOptimizer`.
  - `grit/manager.py`: Contains the `GRITManager` for handling GRIT-specific operations.
  - `grit/config.py`: The GRIT configuration.
  - `grit/autograd.py`: Custom autograd functions for GRIT.

- **Replaced Custom Trainer with `GritTrainer` Class**: The previous functional training loop in `training/trainer.py` was replaced by a more robust, class-based `GritTrainer` that inherits from the Hugging Face `Seq2SeqTrainer`. This change provides:
  - **Encapsulation**: The entire training and evaluation loop is now self-contained.
  - **Extensibility**: Easier to add callbacks, logging, and other features.
  - **Consistency**: Aligns with the standard Hugging Face training paradigm.

- **Integrated `GritOptimizer` and `GRITManager`**: The `main.py` script was refactored to replace the standard `AdamW` optimizer with the `GritOptimizer` and to instantiate the `GRITManager`. This ensures that the core GRIT preconditioning and regularization logic is correctly applied during training.

- **Centralized `main.py` Entry Point**: The `main.py` script is now a clean, high-level entry point. Its sole responsibilities are to:
  1. Initialize the configuration, model, and multimodal dataset.
  2. Instantiate the `GRITManager` and `GritTrainer`.
  3. Start the training process by calling `trainer.train()`.
  This delegates all complex training logic to the `GritTrainer` and `GRITManager`, significantly improving readability and separation of concerns.

- **Preserved Multimodal Data Handling**: The most critical constraint was to maintain the existing multimodal data pipeline. This was achieved by ensuring the `compute_loss` method within the new `GritTrainer` correctly handles the dictionary-based batches produced by `VQADataset`. The model's forward pass continues to receive image and text data seamlessly via the `model(**inputs)` pattern.