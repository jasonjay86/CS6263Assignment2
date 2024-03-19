from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
import evaluate

max_length = 256
#test
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=max_length)

modelName = "microsoft/phi-2"
dataset = load_dataset("flytech/python-codes-25k", split='train').train_test_split(test_size=1000,train_size=100)

tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True, torch_dtype=torch.float32)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained(modelName, trust_remote_code=True, torch_dtype=torch.float32)
# model = AutoModelForCausalLM.from_pretrained(modelName, torch_dtype="auto", trust_remote_code=True, torch_dtype=torch.float32)
# model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=4)
training_args = TrainingArguments(output_dir="test_trainer",evaluation_strategy="epoch",label_names=["instruction", "input", "output"])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()