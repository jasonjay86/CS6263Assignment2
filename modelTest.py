from transformers import (
    AutoModelForCausalLM,
     AutoTokenizer,
     pipeline)
# from peft import PeftModel, PeftConfig

modelpath = "./Mistral"
testPrompt = "Write a program to impress my daughter"

model = AutoModelForCausalLM.from_pretrained(modelpath, device_map="auto")
# model = PeftModel.from_pretrained(model, modelpath)

tokenizer = AutoTokenizer.from_pretrained(modelpath)

input = tokenizer(testPrompt, return_tensors="pt").input_ids
input = input.to('cuda')
outputs = model.generate(input, max_length = 500)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# pipe = pipeline("question-answering",model = modelpath)
# print(pipe(question=testPrompt,context ="fibonacci", max_length=200))

print(text)