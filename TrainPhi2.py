import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from tqdm.notebook import tqdm
from trl import SFTTrainer
from huggingface_hub import interpreter_login
from accelerate import Accelerator

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)


accelerator = Accelerator()

# Clear GPU memory
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
    "microsoft/phi-2", 
    device_map=device_map,
    trust_remote_code=True,
)

# Set the pretraining_tp to 1
model.config.pretraining_tp = 1

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Configure the LoraConfig for PEFT
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense',
        'fc1',
        'fc2',
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Get the PEFT model
lora_model = get_peft_model(model, peft_config)

# Prepare the model for training with the Accelerator
lora_model = accelerator.prepare_model(lora_model)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Configure the training arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    prediction_loss_only=True,
)

# Disable cache for the Lora model
lora_model.config.use_cache = False

# Load the dataset
dataset = load_dataset("flytech/python-codes-25k", split='train').train_test_split(test_size=.001,train_size=.01)

# Create the SFTTrainer
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
trainer.save_model("./FTPhi2_dev")
