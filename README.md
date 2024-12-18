# Fine-Tuning Language Models

This README provides instructions for fine-tuning a language model. Follow the steps below to set up your environment, fine-tune the model, and run inference.

---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Code Execution Steps](#code-execution-steps)
   - [Step 1: Install Dependencies](#step-1-install-dependencies)
   - [Step 2: Initialize the Model](#step-2-initialize-the-model)
   - [Step 3: Prepare the Dataset](#step-3-prepare-the-dataset)
   - [Step 4: Fine-Tune the Model](#step-4-fine-tune-the-model)
   - [Step 5: Run Inference](#step-5-run-inference)
3. [Example Output](#example-output)

---

## Environment Setup

Ensure you are using **Google Colab** with GPU acceleration enabled. To enable GPU:
1. Go to the menu and select **Runtime > Change runtime type**.
2. Under "Hardware accelerator," choose **GPU**.
3. Click **Save**.

---

## Code Execution Steps

### Step 1: Install Dependencies

Run the following command in a Colab cell to install the required dependencies:

```bash
!pip install unsloth
```

---

### Step 2: Initialize the Model

Copy and paste the following code into a new cell. This initializes the model with configurations for fine-tuning:

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

class ModelConfig:
    MAX_SEQ_LENGTH = 2048
    DTYPE = None  # None for auto-detection
    LOAD_IN_4BIT = True

class ModelSetup:
    @staticmethod
    def initialize_model(model_name):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
            dtype=ModelConfig.DTYPE,
            load_in_4bit=ModelConfig.LOAD_IN_4BIT,
        )
        return model, tokenizer

    @staticmethod
    def add_lora_adapters(model):
        return FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
```

---

### Step 3: Prepare the Dataset

Add the following code to a new cell to prepare the dataset for training:

```python
class DataPreparation:
    @staticmethod
    def prepare_dataset():
        alpaca_prompt = (
            """Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"""
            """### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"""
        )

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
                texts.append(text)
            return {"text": texts}

        dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        dataset = dataset.map(formatting_prompts_func, batched=True)
        return dataset
```

---

### Step 4: Fine-Tune the Model

Run the following code in a new cell to fine-tune the model:

```python
class Training:
    @staticmethod
    def train_model(model, tokenizer, dataset):
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False,
            args=training_args,
        )

        return trainer.train()
```

---

### Step 5: Run Inference

Use the following code to run inference on the fine-tuned model:

```python
class Inference:
    @staticmethod
    def run_inference(model, tokenizer, instruction, input_text):
        alpaca_prompt = (
            """Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"""
            """### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"""
        )

        FastLanguageModel.for_inference(model)

        inputs = tokenizer([
            alpaca_prompt.format(instruction, input_text, "")
        ], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, use_cache=True)
        return tokenizer.batch_decode(outputs)

# Example Usage
if __name__ == "__main__":
    model_name = "unsloth/Qwen2.5-Coder-7B"
    model, tokenizer = ModelSetup.initialize_model(model_name)
    model = ModelSetup.add_lora_adapters(model)

    dataset = DataPreparation.prepare_dataset()
    Training.train_model(model, tokenizer, dataset)

    results = Inference.run_inference(
        model, tokenizer, "Write a function for a merge interval and show the command I need to run in Python.", ""
    )
    print(results[0])
```

---

## Example Output

After running the code, you can expect output similar to the following:

```plaintext
Step    Training Loss
10      0.807900
20      0.571600
30      0.585300
40      0.546300
50      0.471200
60      0.529000

# Below is an instruction that describes a task, paired with an input that provides further context.
# Write a response that appropriately completes the request.

### Instruction:
Write a function for a merge interval and show the command I need to run in Python.

### Input:

### Response:
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

# Command to run in Python
print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))
```

_Leverage Google Colab for this code._
