# CS6263 Assignment2 - Jason Johnson
## Instructions
### Envrionment Setup
To run, first load the environment from the environment.yml file with:

`conda env create -f environment.yml`

Then activate it:

`conda activate assignment2`

**1)	We would like for you to present and visualize the probabilities of each token in the vocabulary from early exit layers (premature vocabulary distribution layers) vs. mature layer (last layer – Layer 32).**

### Layer 8
![token_probability_layer_8](https://github.com/jasonjay86/CS6263Assignment2/assets/65077765/055407ad-55f1-4dba-a82e-8bf99132dc2f)

### Layer 16
![token_probability_layer_16](https://github.com/jasonjay86/CS6263Assignment2/assets/65077765/eb5b486e-d989-4ff9-827f-1fd7d38f2ec5)

### Layer 24
![token_probability_layer_24](https://github.com/jasonjay86/CS6263Assignment2/assets/65077765/aa0ec2c4-4f6e-4c2f-b745-110598cca5d8)

### Layer 32
![token_probability_layer_32](https://github.com/jasonjay86/CS6263Assignment2/assets/65077765/808085e3-def3-4cb9-b7f3-66e720d712db)

**2)	If you recall the paper we reviews on consistency checking used several models, do you think we can use consistency check method between these layers for factuality analysis? Present your approach and results including discussion.**
Yes I believe consistency checking would be somewhat effective by looking at the different layers.  Intuitively, if a token had a high probability in all observered layers, one could expect that that token is solid and not being hallucinated.  The token would have "survived" so many layers, meaning that the model always had high confidence in it.
