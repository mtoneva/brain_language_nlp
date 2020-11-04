# Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)

This repository contains code for the paper [Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)](https://arxiv.org/pdf/1905.11833.pdf)

Bibtex: 
```
@inproceedings{toneva2019interpreting,
  title={Interpreting and improving natural-language processing (in machines) with natural language-processing (in the brain)},
  author={Toneva, Mariya and Wehbe, Leila},
  booktitle={Advances in Neural Information Processing Systems},
  pages={14954--14964},
  year={2019}
}
```
## fMRI Recordings of 8 Subjects Reading Harry Potter
You can download the already [preprocessed data here](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8?usp=sharing). This data contains fMRI recordings for 8 subjects reading one chapter of Harry Potter. The data been detrended, smoothed, and trimmed to remove the first 20TRs and the last 15TRs. For more information about the data, refer to the paper. We have also provided the precomputed voxel neighborhoods that we have used to compute the searchlight classification accuracies. 

The following code expects that these directories are positioned under the data folder in this repository (e.g. `./data/fMRI/` and `./data/voxel_neighborhoods`.


## Measuring Alignment Between Brain Recordings and NLP representations

Our approach consists of three main steps:
1. Derive representations of text from an NLP model
2. Build an encoding model that takes the derived NLP representations as input and predicts brain recordings of people reading the same text
3. Evaluates the predictions of the encoding model using a classification task

In our paper, we present alignment results from 4 different NLP models - ELMo, BERT, Transformer-XL, and USE. Below we provide an overview of how to run all three steps.


### Deriving representations of text from an NLP model

Needed dependencies for each model:
- USE: Tensorflow < 1.8,  `pip install tensorflow_hub`
- ELMo: `pip install allennlp`
- BERT/Transformer-XL: `pip install pytorch_pretrained_bert`


The following command can be used to derive the NLP features that we used to obtain the results in Figures 2 and 3:
```
python extract_nlp_features.py
    --nlp_model [bert/transformer_xl/elmo/use]   
    --sequence_length s
    --output_dir nlp_features
```
where s ranges from to 1 to 40. This command derives the representation for all sequences of `s` consecutive words in the stimuli text in `/data/stimuli_words.npy` from the model specified in `--nlp_model` and saves one file for each layer in the model in the specified `--output_dir`. The names of the saved files contain the argument values that were used to generate them. The output files are numpy arrays of size `n_words x n_dimensions`, where `n_words` is the number of words in the stimulus text and `n_dimensions` is the number of dimensions in the embeddings of the specified model in `--nlp_model`. Each row of the output file contains the representation of the most recent `s` consecutive words in the stimulus text (i.e. row `i` of the output file is derived by passing words `i-s+1` to `i` through the pretrained NLP model).


### Building encoding model to predict fMRI recordings

Note: This code has been tested using python3.7

```
python predict_brain_from_nlp.py
    --subject [F,H,I,J,K,L,M,N]
    --nlp_feat_type [bert/elmo/transformer_xl/use]   
    --nlp_feat_dir INPUT_FEAT_DIR
    --layer l
    --sequence_length s
    --output_dir OUTPUT_DIR
```

This call builds encoding models to predict the fMRI recordings using representations of the text stimuli derived from NLP models in step 1 above (`INPUT_FEAT_DIR` is set to the same directory where the NLP features from step 1 were saved, `l` and `s` are the layer and sequence length to be used to load the extracted NLP representations). The encoding model is trained using ridge regression and 4-fold cross validation. The predictions of the encoding model for the heldout data in every fold are saved in an output file in the specified directory `OUTPUT_DIR`. The output filename is in the following format: `predict_{}_with_{}_layer_{}_len_{}.npy`, where the first field is specified by `--subject`, the second by `--nlp_feat_type`, and the rest by `--layer` and `--sequence_length`.

### Evaluating the predictions of the encoding model using classification accuracy

Note: This code has been tested using python3.7

```
python evaluate_brain_predictions.py
    --input_path INPUT_PATH
    --output_path OUTPUT_PATH
    --subject [F,H,I,J,K,L,M,N]
```

This call computes the mean 20v20 classification accuracy (over 1000 samplings of 20 words) for each encoding model (from each of the 4 CV folds). The output is a `pickle` file that contains a list with 4 elements -- one for each CV fold. Each of these 4 elements is another list, which contains the accuracies for all voxels. `INPUT_PATH` is the full path (including the file name) to the predictions saved in step 2 above. `OUTPUT_PATH` is the complete path (including file name) to where the accuracies should be saved. 

The following extracts the average accuracy across CV folds for a particular subject:
```
import pickle as pk
import numpy as np
loaded = pk.load(open('{}_accs.pkl'.format(OUTPUT_PATH), 'rb'))
mean_subj_acc_across_folds = loaded.mean(0)
```
