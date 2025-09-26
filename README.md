# Grit Multimodal Finetuning

This repository provides a framework for fine-tuning multimodal models using the **GRIT (Gradient-based Riemannian-tangent Tuning)** method and **LoRA (Low-Rank Adaptation)**. The code is designed to work with the `Qwen2-VL` series of models and is demonstrated on a Visual Question Answering (VQA) task.

## Features

*   **GRIT Fine-tuning:** Implements the GRIT method for efficient and effective fine-tuning of large multimodal models.
*   **LoRA Fine-tuning:** Includes a standard LoRA implementation for comparison and as a baseline.
*   **Qwen2-VL Support:** Natively supports the `Qwen/Qwen2-VL-7B-Instruct` and `Qwen/Qwen2-VL-2B-Instruct` models.
*   **VQA Task:** Provides a `VQADataset` class and training loops for the VQA task.
*   **K-FAC Integration:** Utilizes K-FAC (Kronecker-Factored Approximate Curvature) for second-order optimization in the GRIT method.
*   **Modular Design:** The code is organized into modules for easy understanding and extension.

## Requirements

The project requires the following Python libraries:

*   `torch`
*   `transformers`
*   `pillow`
*   `tqdm`
*   `numpy`
*   `huggingface-hub`
*   `scikit-learn`
*   `peft`

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

The repository provides two main training scripts: `main.py` for GRIT fine-tuning and `lora_main.py` for LoRA fine-tuning.

### GRIT Fine-tuning

To start the GRIT fine-tuning, run the following command:

```bash
python -m main
```

The training process will be configured by the `GRITConfig` class in `config/config.py`.

### LoRA Fine-tuning

To start the LoRA fine-tuning, run the following command:

```bash
python -m lora_main
```

The training process will be configured by the `LoRAConfig` class in `config/lora_config.py`.

## Configuration

The hyperparameters and training settings can be modified in the following files:

*   **GRIT Configuration:** `config/config.py`
*   **LoRA Configuration:** `config/lora_config.py`

These files contain settings for the model, training parameters, and method-specific hyperparameters.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
