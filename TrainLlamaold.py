import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from tqdm.notebook import tqdm

from trl import SFTTrainer
from huggingface_hub import interpreter_login
from accelerate import Accelerator

# Set the path to the model
modelpath = "meta-llama/Llama-2-7b-hf"

# Initialize the accelerator
accelerator = Accelerator()

# Login to the Hugging Face interpreter
interpreter_login()

# Clear CUDA cache
torch.cuda.empty_cache()

# Set the compute data type to float16
compute_dtype = getattr(torch, "float16")

# Configure BitsAndBytes quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant=False,
)

# Set the device map to "auto"
device_map = "auto"

# Download the model
model = AutoModelForCausalLM.from_pretrained(
    modelpath, 
    # quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    use_auth_token=True
)

# Set the pretraining_tp attribute of the model's config to 1
model.config.pretraining_tp = 1

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Configure LoraConfig for PEFT
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Get the PEFT model
lora_model = get_peft_model(model, peft_config)

# Prepare the model using the accelerator
lora_model = accelerator.prepare_model(lora_model)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Configure the training arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    prediction_loss_only=True,
    # gradient_accumulation_steps=4,
    # optim="paged_adamw_32bit",
    # save_steps=500,
    # logging_steps=10,
    # learning_rate=2e-4,
    # fp16=False,
    # bf16=True,
    # max_grad_norm=.3,
    # max_steps=10000,
    # warmup_ratio=.03,
    # group_by_length=True,
    # lr_scheduler_type="constant",
)

# Disable cache usage in the model's config
lora_model.config.use_cache = False

# Load the dataset
dataset = load_dataset("flytech/python-codes-25k", split='train').train_test_split(test_size=.001, train_size=.01)

# Initialize the trainer
trainer = SFTTrainer(
    model=lora_model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./Llama")