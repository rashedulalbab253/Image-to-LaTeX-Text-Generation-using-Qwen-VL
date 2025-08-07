# Fine-Tuning Qwen2-VL-7B-Instruct for LaTeX OCR with Unsloth

This project demonstrates how to fine-tune the [Qwen2-VL-7B-Instruct](https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct) vision-language model using the [Unsloth](https://github.com/unslothai/unsloth) library for the task of LaTeX Optical Character Recognition (OCR). The goal is to enable the model to generate accurate LaTeX code from images containing mathematical formulas.

## ⚠️ Note

> My previous GitHub account was unexpectedly suspended. This project was originally created earlier and has been re-uploaded here. All work was done gradually over time, and original commit history has been preserved where possible.

---

## Features

- **Efficient Fine-Tuning:** Utilizes 4-bit quantization and LoRA for parameter-efficient training.
- **Multi-Modal Data:** Works with datasets containing both images and their LaTeX representations.
- **Reproducible Pipeline:** End-to-end workflow from data loading to model evaluation.
- **Fast Training:** Leverages Unsloth's optimizations for rapid experimentation on GPUs.

## Workflow Overview

1. **Install Dependencies:** All required libraries (Unsloth, bitsandbytes, TRL, datasets, etc.) are installed in the notebook.
2. **Load Pretrained Model:** Qwen2-VL-7B-Instruct is loaded with 4-bit quantization.
3. **PEFT Configuration:** LoRA is applied for efficient fine-tuning on both vision and language layers.
4. **Prepare Dataset:** Loads the [unsloth/Latex_OCR](https://huggingface.co/datasets/unsloth/Latex_OCR) dataset, containing images and corresponding LaTeX texts.
5. **Format as Conversation:** Data samples are converted into a chat-like format suitable for vision-language instruction tuning.
6. **Fine-Tune:** The model is trained on the formatted data for a demonstration number of steps.
7. **Inference:** The trained model is evaluated on examples, generating LaTeX from images.

## Example Usage

```python
from unsloth import FastVisionModel
from datasets import load_dataset

# Load the model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-7B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)

# Load dataset
dataset = load_dataset("unsloth/Latex_OCR", split="train")

# Convert a sample for chat-based fine-tuning
def convert_to_conversation(sample):
    instruction = "Write the LaTeX representation for this image."
    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": sample["image"]}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": sample["text"]}
        ]}
    ]
    return {"messages": conversation}

# Prepare data for training
converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# ... (see notebook for full training pipeline)
```

## Requirements

- Python 3.8+
- A CUDA-enabled GPU (e.g., NVIDIA T4, A100)
- See `requirements.txt` or the notebook for all Python dependencies.

## References

- [Qwen2-VL-7B-Instruct on HuggingFace](https://huggingface.co/unsloth/Qwen2-VL-7B-Instruct)
- [Unsloth Library](https://github.com/unslothai/unsloth)
- [Latex_OCR Dataset](https://huggingface.co/datasets/unsloth/Latex_OCR)
- [Transformers Library](https://github.com/huggingface/transformers)
- [TRL Library](https://github.com/huggingface/trl)

## License

This project uses open-source models and datasets; consult each library/model/dataset for its specific license.

---

**Author:** [Rashedul Albab]  
**Last updated:** 2025-08-07
