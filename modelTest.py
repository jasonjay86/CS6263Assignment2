from transformers import (
    AutoModelForCausalLM,
     AutoTokenizer,
     pipeline)
# from peft import PeftModel, PeftConfig

modelpath = "./Mistral"
testPrompt = "Develop a Python code to generate the nth number in the Fibonacci series n = 8"

# model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto")
# model = PeftModel.from_pretrained(model, modelpath)

# tokenizer = AutoTokenizer.from_pretrained(modelpath)

# input = tokenizer(testPrompt, return_tensors="pt").input_ids
# outputs = model.generate(input, max_length = 200)
# text = tokenizer.decode(outputs[0], skip_special_tokens=True)

pipe = pipeline("question-answering",model = modelpath,device = 0)
print(pipe(question=testPrompt,context ="fibonacci!", max_length=200))

# print(text)