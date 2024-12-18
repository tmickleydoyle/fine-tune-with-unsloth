# Fine-Tuning with Unsloth

This project demonstrates fine-tuning and inference using the `Qwen2.5-Coder-7B` model with the `unsloth` library.

## Setup

1. **Install required libraries**:
   ```bash
   pip install unsloth transformers datasets trl peft
   ```
2. **Authenticate Hugging Face**:
   Set your Hugging Face token:
   ```python
   from google.colab import userdata
   HF_TOKEN = userdata.get('HF_TOKEN')
   ```

## Steps

### 1. Initialize the Model
The script loads the model and tokenizer with LoRA adapters for efficient fine-tuning.

### 2. Prepare the Dataset
It uses an Alpaca-style dataset for training (`iamtarun/python_code_instructions_18k_alpaca`).

### 3. Train the Model
The fine-tuning process is configured to run for 2 epochs with a batch size of 2.

### 4. Save and Share
After training:
- Save the model locally.
- Optionally push it to the Hugging Face Hub.

### 5. Run Inference
Use the model to generate responses for custom instructions. Example:
```python
results = Inference.run_inference(
    model, tokenizer, 
    "Write a function for a merge interval.", ""
)
print(results[0])
```

## Notes
- Ensure CUDA is enabled for better performance.
- Update `ORG_NAME` and `MODEL` to use your Hugging Face repository.

That's it! You can now fine-tune and use `Qwen2.5-Coder-7B` for your tasks.
