# Minimalist Transformer for Sequence-to-Sequence:

This repo is a Minimalist transformer for sequence to sequence modelling.

The task explored in this repo is the task of inverting sequences with a transformer model.

For learning abouth transformers I recommend the following blog posts:
- [Transformers from scratch](http://www.peterbloem.nl/blog/transformers) by Peter Bloem 
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar

To see a Minimalist implementation of a transformer for sentiment analysis check out my other repo: 
- [Minimalist-Transformer-Classifier](https://github.com/ricardorei/Minimalist-Transformer-Classifier)

### Requirements:

This project uses Python 3.6.

Create a virtual env with:
```sh 
virtualenv -p python3.6 env
```
Activate venv:
```sh 
source env/bin/activate
```

Finally, to install all requirements just run:
```sh 
pip install -r requirements.txt
```

### Preprocessing:

Before we start is important to preprocess our data and create our vocabulary.

Run the following command:
```sh 
python transformer preprocess
```

It will print a dictionary, our vocabulary string to index.

### Train:

To train the model run:
```sh 
python transformer train
```

This command will print the losses along the training and validation sets and 
ends by running the model with some final source samples and using greedy search to decode the reverted sequence.
