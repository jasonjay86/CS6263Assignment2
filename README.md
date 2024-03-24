# CS6263 Assignment1C - Jason Johnson
## Instructions
### Envrionment Setup
To run, first load the environment from the environment.yml file with:

`conda env create -f environment.yml`

Then activate it:

`conda activate arc1c`

### Fine Tuning (Task 1)
If that is successful, you can fine tune Llama, Mistral, or Phi2 on [the python code dataset](https://huggingface.co/datasets/flytech/python-codes-25k) with:

`python TrainLlama.py`

or

`python TrainMistral.py`

or

`python TrainPhi2.py`

**NOTE** I had trouble getting the code to work on a GPU, even on the ARC.  So training is *extremely* slow...Several hours for about 500 rows of the dataset.

### Metric Evaluations
Open `Evaluation.py` and comment/uncomment the models, the metrics to capture, and the hyperparamers and sizes to execute with.  All of this is in the beginning of the code, easy to find. There is also a variable to set the number of rows to execute.

**NOTE** Be sure that the fine tuned model is in the current directory to run

**LAST NOTE**  Evaluation.py is also very slow

Run `Evaluation.py` with

`python Evaluation.py`

**Please see metrics tables and task discussions below**

## Task 2
![image](https://github.com/jasonjay86/CS6263Assignment1C/assets/65077765/30b5b9fb-736f-43f0-8fd0-26e58f8f736f)

**Write a discussion (4-5 Lines) explaining the comparison between two models. Moreover, compare the metrics and discuss which metrics are more appropriate compared to human evaluation.**
Mistral and Phi-2 edgeds out Llama in most metrics.  Phi-2 performed especially well. Llama lost points  because it often added answers to instructions it was not given.  It had a hard time knowing when the answer was complete.  This might have been because of the limited training I had to do.  Phi-2 and Mistral also had this problem but not to the degree that LLaMA did.  It appears the the CodeBleu score most closely corresponds to the the Human Evaluation score.  Human evaluation scores are roughly double the codeBleu scores across the board.  I believe that the ratio is important and the lower scores on the codeBleu metric are do to the models' tendencies to explain their work in uncompilable text.  This uncompilable text probably lowered scores for all three models.

## Task 3
![image](https://github.com/jasonjay86/CS6263Assignment1C/assets/65077765/ee2ceeed-5807-43b5-b164-696956a2235e)![image](https://github.com/jasonjay86/CS6263Assignment1C/assets/65077765/19beecaa-94de-49a4-92ea-2c8afe2cc77b)
**Write another discussion explaining the how the hyperparameters effect on the different metrics of LLaMA and Phi-2 (4-5 Lines).**
Increasing the top_k parameter appears to increase the human evaluation scores for each model.  Thi i likely because it opens up the model to select from a larger smapling of next tokens.  Increasing this value further may improve scores even higher.  Increasing the beam size hyperparameter also appears to raise scores across the board for every metric except for the human evaluation.  This may be do the algorithmic metrics focusing on matching the reference texts while human evaluation values working code.  Finally Temparature seems to worsen the outputs as it increases.  There appears to be a "sweet spot" .25 and .5.  Having temperature too high opens the door for low probability tokens that may be the difference between code that compiles and code that does not.
