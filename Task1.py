from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

modelName = "microsoft/phi-2"
dataset = load_dataset("flytech/python-codes-25k")

tokenizer = AutoTokenizer.from_pretrained(modelName)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=4)
training_args = TrainingArguments(output_dir="test_trainer",evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"][0:100],
    eval_dataset=tokenized_datasets["train"][101:200],
    compute_metrics=compute_metrics,
)

trainer.train()