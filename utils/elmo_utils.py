from allennlp.commands.elmo import ElmoEmbedder
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer

import numpy as np
import torch
import time as tm

def get_elmo_layer_representations(seq_len, text_array, remove_chars, word_ind_to_extract):

    model = ElmoEmbedder()
    tokenizer = WordTokenizer()
    
    # where to store layer-wise elmo embeddings of particular length
    elmo = {}
    for layer in range(-1,3):
        elmo[layer] = []

    if word_ind_to_extract < 0: # the index is specified from the end of the array, so invert the index
        from_start_word_ind_to_extract = seq_len + word_ind_to_extract
    else:
        from_start_word_ind_to_extract = word_ind_to_extract

    start_time = tm.time()    
        
    # before we've seen enough words to make up the sequence length, add the representation for the last word 'seq_len' times
    word_seq = text_array[:seq_len]
    for _ in range(seq_len):
        elmo = add_avrg_token_embedding_for_specific_word(word_seq,
                                                                     tokenizer,
                                                                     model,
                                                                     remove_chars,
                                                                     from_start_word_ind_to_extract,
                                                                     elmo)

    # then add the embedding of the last word in a sequence as the embedding for the sequence
    for end_curr_seq in range(seq_len, len(text_array)):
        word_seq = text_array[end_curr_seq-seq_len+1:end_curr_seq+1]
        elmo = add_avrg_token_embedding_for_specific_word(word_seq,
                                                          tokenizer,
                                                          model,
                                                          remove_chars,
                                                          from_start_word_ind_to_extract,
                                                          elmo)

        if end_curr_seq % 100 == 0:
            print('Completed {} out of {}: {}'.format(end_curr_seq, len(text_array), tm.time()-start_time))
            start_time = tm.time()

    print('Done extracting sequences of length {}'.format(seq_len))
    
    return elmo 


def predict_elmo_embeddings(words_in_array, tokenizer, model, remove_chars):    
    
    n_seq_tokens = 0
    seq_tokens = []
    
    word_ind_to_token_ind = {}             # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    
    for i,word in enumerate(words_in_array):
        word_ind_to_token_ind[i] = []      # initialize token indices array for current word

        word_tokens = tokenizer.tokenize(word)
            
        for token in word_tokens:
            if token not in remove_chars:  # don't add any tokens that are in remove_chars
                seq_tokens.append(token)
                word_ind_to_token_ind[i].append(n_seq_tokens)
                n_seq_tokens = n_seq_tokens + 1
    
    
    encoded_layers = model.embed_sentence(seq_tokens)
    
    return encoded_layers, word_ind_to_token_ind


# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
#
# word_seq: numpy array of words in input sequence
# remove_chars: characters that should not be included in the represention when word_seq is tokenized
# from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
# model_dict: where to save the extracted embeddings
def add_avrg_token_embedding_for_specific_word(word_seq,tokenizer,model,remove_chars,from_start_word_ind_to_extract,model_dict):
    
    word_seq = list(word_seq)  
    all_sequence_embeddings, word_ind_to_token_ind, _ = predict_elmo_embeddings(word_seq, tokenizer, model, remove_chars)
    token_inds_to_avrg = word_ind_to_token_ind[from_start_word_ind_to_extract]
    model_dict = add_word_elmo_embedding(model_dict, all_sequence_embeddings,token_inds_to_avrg)
    
    return model_dict

# add the embeddings for a specific word in the sequence
# token_inds_to_avrg: indices of tokens in embeddings output to avrg
def add_word_elmo_embedding(elmo_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):
    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.detach().numpy()
        elmo_dict[specific_layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0))
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.detach().numpy()
            elmo_dict[layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0)) # avrg over all tokens for specified word
    return elmo_dict


    
# add the embeddings for all individual words
def add_all_elmo_embeddings(elmo_dict, embeddings_to_add):
    for layer in range(3):
      
        seq_len = embeddings_to_add.shape[1]
        
        for word in range(seq_len):
            elmo_dict[layer].append(embeddings_to_add[layer,word,:])
    return elmo_dict
        
# add the embeddings for only the last word in the sequence
def add_last_elmo_embedding(elmo_dict, embeddings_to_add):
    for layer in range(3):        
        elmo_dict[layer].append(embeddings_to_add[layer,-1,:])
    return elmo_dict