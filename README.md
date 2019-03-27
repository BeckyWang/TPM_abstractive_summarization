# TPM_abstractive_summarization
A Topic-aware Pointer Model for Abstractive Summarization using Topic Relevance Loss

## Looking for baseline model
This code is based on the [pointer-generator model](https://github.com/abisee/pointer-generator)

**Note** Baseline code is in Python 2. If you want a Python 3 version, see [@becxer's fork](https://github.com/becxer/pointer-generator/).

## How to get training/test dataset
This code can be used in Chinese and English datasets. 

English datasets-CNN/Daily Mail, Instructions are [here](https://github.com/abisee/cnn-dailymail)

## Data preposessing
In our work, we firstly use the pre-trained topic model to get the topic information, then integrating it into the model from two aspects: pointer mechanism and attention mechanism. 

The details of data preprocessing are available [here](https://github.com/BeckyWang/data_preprocessing_for_TPM).

## How to run
Follow the instructions of [pointer-generator model](https://github.com/abisee/pointer-generator) 

## Evaluate with ROUGE
If you are using an English dataset, you can use the Python package [`pyrouge`](https://pypi.python.org/pypi/pyrouge) to run ROUGE evaluation. Some useful instructions in [pointer-generator model](https://github.com/abisee/pointer-generator).

If you are using an Chinese dataset, just follow this code. We have implemented the ROUGE method ourselves in `run_rouge.py`.