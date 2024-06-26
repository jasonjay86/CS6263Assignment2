
import os
from dataclasses import dataclass, field
from typing import Optional
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from tqdm.notebook import tqdm

from trl import SFTTrainer
from accelerate import Accelerator

accelerator = Accelerator()

# interpreter_login()

torch.cuda.empty_cache() 

# compute_dtype = getattr(torch, "float16")

device_map = "auto"
max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
# Download model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-2-7b-hf", # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

print(model)

# model.config.pretraining_tp = 1

# Enable gradient checkpointing to save memory
# model.gradient_checkpointing_enable()

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# # Get the PEFT model
# lora_model = get_peft_model(model, peft_config)

# # Prepare the model for training
# lora_model = accelerator.prepare_model(lora_model)

# # Create the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

# # Define the training arguments
# training_arguments = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     prediction_loss_only=True,
#     # gradient_accumulation_steps=4,
#     # optim="paged_adamw_32bit",
#     # save_steps=500, #CHANGE THIS IF YOU WANT IT TO SAVE LESS OFTEN. I WOULDN'T SAVE MORE OFTEN BECAUSE OF SPACE
#     # logging_steps=10,
#     # learning_rate=2e-4,
#     # fp16=False,
#     # bf16=True,
#     # max_grad_norm=.3,
#     # max_steps=10000,
#     # warmup_ratio=.03,
#     # group_by_length=True,
#     # lr_scheduler_type="constant",
# )

# Disable cache in the model config
# lora_model.config.use_cache = False

# Load the dataset
dataset = load_dataset("flytech/python-codes-25k", split='train').train_test_split(test_size=.8, train_size=.2)

# Create the trainer
trainer = SFTTrainer(
    model = model,
    # train_dataset = dataset,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 1000,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 100,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 1222,
    ),
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./Llama")