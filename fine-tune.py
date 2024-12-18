# !pip install unsloth

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

class Training:
    @staticmethod
    def train_model(model, tokenizer, dataset):
        training_args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            # num_train_epochs=2,
            max_steps = 60,
            learning_rate=2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
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
