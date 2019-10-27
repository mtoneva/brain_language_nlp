# Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)

This repository contains code for the paper [Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)](https://arxiv.org/pdf/1905.11833.pdf)

Bibtex: 
```
@inproceedings{brain_language_nlp,
    title={Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)},
    author={Toneva, Mariya and Wehbe, Leila},
    booktitle={NeurIPS},
    year={2019}
}
```

## Measuring Alignment Between Brain Recordings and NLP representations

Our approach consists of three main steps:
1. Derive representations of text from an NLP model
2. Build an encoding model that takes the derived NLP representations as input and predicts brain recordings of people reading the same text
3. Evaluates the predictions of the encoding model using a classification task

In our paper, we present alignment results from 4 different NLP models - ELMo, BERT, Transformer-XL, and USE. Below we provide an overview of how to derive the representations from these NLP models, and we will soon provide code for steps 2 and 3.


### Deriving representations of text from an NLP model

Needed dependencies for each model:
- USE: Tensorflow < 1.8
- ELMo: `pip install allennlp`
- BERT/Transformer-XL: `pip install pytorch_pretrained_bert`


The following command can be used to derive the NLP features that we used to obtain the results in Figures 2 and 3:
```
python extract_nlp_features.py
    --nlp_model [bert/transformer_xl/elmo/use]   
    --sequence_length s
    --output_dir nlp_features
```
where s ranges from to 1 to 40. This command derives the representation for all sequences of `s` consecutive words in the stimuli text in `/data/stimuli_words.npy` from the model specified in `--nlp_model` and saves one file for each layer in the model in the specified `--output_dir`. The names of the saved files contain the argument values that were used to generate them. The output files are numpy arrays of size `n_words x n_dimensions`, where `n_words` is the number of words in the stimulus text and `n_dimensions` is the number of dimensions in the embeddings of the specified model in `--nlp_model`. Each row of the output file contains the representation of the most recent `s` consecutive words in the stimulus text.
