from transformers import (
    AutoModelForCausalLM,
     AutoTokenizer)
from peft import PeftModel, PeftConfig

modelpath = "microsoft/phi-2"
testPrompt = "Develop a Python code to generate the nth number in the Fibonacci series n = 8"
model = AutoModelForCausalLM.from_pretrained(modelpath)
# model = PeftModel.from_pretrained(model, modelpath)
# model = AutoModelForCausalLM.from_pretrained(modelpath)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

input = tokenizer(testPrompt, return_tensors="pt").input_ids
outputs = model.generate(input, max_length = 200)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text)