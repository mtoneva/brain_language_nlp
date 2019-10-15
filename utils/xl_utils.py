import numpy as np
import torch
import time as tm

from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel


def get_xl_layer_representations(seq_len, text_array, remove_chars, word_ind_to_extract):

    model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    model.eval()
    

    # get the token embeddings
    token_embeddings = []
    for word in text_array:
        current_token_embedding = get_xl_token_embeddings([word], tokenizer, model, remove_chars)
        token_embeddings.append(np.mean(current_token_embedding.detach().numpy(), 1))
    
    # where to store layer-wise xl embeddings of particular length
    XL = {}
    for layer in range(19):
        XL[layer] = []
    XL[-1] = token_embeddings

    if word_ind_to_extract < 0: # the index is specified from the end of the array, so invert the index
        from_start_word_ind_to_extract = seq_len + word_ind_to_extract  
    else:
        from_start_word_ind_to_extract = word_ind_to_extract

    start_time = tm.time()    
        
    # before we've seen enough words to make up the sequence length, add the representation for the last word 'seq_len' times
    word_seq = text_array[:seq_len]
    for _ in range(seq_len):
        XL = add_avrg_token_embedding_for_specific_word(word_seq,
                                                                     tokenizer,
                                                                     model,
                                                                     remove_chars,
                                                                     from_start_word_ind_to_extract,
                                                                     XL)

    # then add the embedding of the last word in a sequence as the embedding for the sequence
    for end_curr_seq in range(seq_len, len(text_array)):
        word_seq = text_array[end_curr_seq-seq_len+1:end_curr_seq+1]
        XL = add_avrg_token_embedding_for_specific_word(word_seq,
                                                          tokenizer,
                                                          model,
                                                          remove_chars,
                                                          from_start_word_ind_to_extract,
                                                          XL)

        if end_curr_seq % 100 == 0:
            print('Completed {} out of {}: {}'.format(end_curr_seq, len(text_array), tm.time()-start_time))
            start_time = tm.time()

    print('Done extracting sequences of length {}'.format(seq_len))
    
    return XL

def predict_xl_embeddings(words_in_array, tokenizer, model, remove_chars):        
    for word in words_in_array:
        if word in remove_chars:
            print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
            return -1
    
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
    
    # convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    
    hidden_states, mems = model(tokens_tensor)
    seq_length = hidden_states.size(1)
    lower_hidden_states = list(t[-seq_length:, ...].transpose(0, 1) for t in mems)
    all_hidden_states = lower_hidden_states + [hidden_states]
    
    return all_hidden_states, word_ind_to_token_ind

# get the XL token embeddings
def get_xl_token_embeddings(words_in_array, tokenizer, model, remove_chars):    
    for word in words_in_array:
        if word in remove_chars:
            print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')
            return -1
    
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
            
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(seq_tokens)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    
    token_embeddings = model.word_emb.forward(tokens_tensor)
    
    return token_embeddings
    
# predicts representations for specific word in input word sequence, and adds to existing layer-wise dictionary
#
# word_seq: numpy array of words in input sequence
# remove_chars: characters that should not be included in the represention when word_seq is tokenized
# from_start_word_ind_to_extract: the index of the word whose features to extract, INDEXED FROM START OF WORD_SEQ
# model_dict: where to save the extracted embeddings
def add_avrg_token_embedding_for_specific_word(word_seq,tokenizer,model,remove_chars,from_start_word_ind_to_extract,model_dict):
    
    word_seq = list(word_seq)  
    all_sequence_embeddings, word_ind_to_token_ind = predict_xl_embeddings(word_seq, tokenizer, model, remove_chars)
    token_inds_to_avrg = word_ind_to_token_ind[from_start_word_ind_to_extract]
    model_dict = add_word_xl_embedding(model_dict, all_sequence_embeddings,token_inds_to_avrg)
    
    return model_dict

# add the embeddings for a specific word in the sequence
# token_inds_to_avrg: indices of tokens in embeddings output to avrg
def add_word_xl_embedding(model_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):
    if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add[specific_layer]
        full_sequence_embedding = layer_embedding.detach().numpy()
        model_dict[specific_layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0))
    else:
        for layer, layer_embedding in enumerate(embeddings_to_add):
            full_sequence_embedding = layer_embedding.detach().numpy()
            model_dict[layer].append(np.mean(full_sequence_embedding[0,token_inds_to_avrg,:],0)) # avrg over all tokens for specified word
    return model_dict
