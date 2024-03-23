# CS6263 Assignment1C - Jason Johnson
## Instructions/Task1
run run run

## Task 2
![image](https://github.com/jasonjay86/CS6263Assignment1C/assets/65077765/30b5b9fb-736f-43f0-8fd0-26e58f8f736f)

**Write a discussion (4-5 Lines) explaining the comparison between two models. Moreover, compare the metrics and discuss which metrics are more appropriate compared to human evaluation.**
Mistral and Phi-2 edgeds out Llama in most metrics.  Phi-2 performed especially well. Llama lost points  because it often added answers to instructions it was not given.  It had a hard time knowing when the answer was complete.  This might have been because of the limited training I had to do.  Phi-2 and Mistral also had this problem but not to the degree that LLaMA did.  It appears the the CodeBleu score most closely corresponds to the the Human Evaluation score.  Human evaluation scores are roughly double the codeBleu scores across the board.  I believe that the ratio is important and the lower scores on the codeBleu metric are do to the models' tendencies to explain their work in uncompilable text.  This uncompilable text probably lowered scores across the board.
