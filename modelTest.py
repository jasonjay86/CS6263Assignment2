from transformers import (
    AutoModelForCausalLM,
     AutoTokenizer)
from peft import PeftModel, PeftConfig

modelpath = "./FTPhi2_dev"
testPrompt = "Edit the following Python program to implement try and except a = 10 b = 0 c = a/b"
model = AutoModelForCausalLM.from_pretrained(modelpath)
model = PeftModel.from_pretrained(model, modelpath)
# model = AutoModelForCausalLM.from_pretrained(modelpath)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

input = tokenizer(testPrompt, return_tensors="pt").input_ids
outputs = model.generate(input, max_length = 100)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text)