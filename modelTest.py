from transformers import (
    AutoModelForCausalLM,
     AutoTokenizer)

modelpath = "./results"
testPrompt = "Write some code to count to 5"

model = AutoModelForCausalLM.from_pretrained(modelpath)
tokenizer = AutoTokenizer.from_pretrained(modelpath)

input = tokenizer(testPrompt, return_tensors="pt").input_ids
outputs = model.generate(input, max_length = 200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
