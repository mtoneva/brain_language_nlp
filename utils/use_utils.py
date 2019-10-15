import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def clean_word(word, remove_chars):
    word2 = word[:]
    while len(word2)>0 and word2[0] in remove_chars:
        word2 = word2[1:]
    while len(word2)>0 and word2[-1] in remove_chars:
        word2 = word2[:-1]    
    return word2

def get_use_layer_representations(seq_len, text_array, remove_chars):
    
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)
    
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    clean_text_array = [clean_word(w,remove_chars) for w in text_array]
    n_labels = len(clean_text_array)

    seq_strings = [" ".join(clean_text_array[i-seq_len:i]) for i in range(20,n_labels)]

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        embeddings = session.run(embed(seq_strings))
        sequence = np.array(embeddings)

    USE = {}
    USE[-1] = [np.zeros((20,sequence.shape[1])),sequence]
    
    return USE
