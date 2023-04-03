import numpy as np
from tqdm import tqdm
import unidecode
import re

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path, encoding="utf8", errors='ignore') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

def build_matrix(word_index, embedding_index):

    """
    https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
    """
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    
    unknown_words = []

    for word, i in tqdm(word_index.items()):
        
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
            continue
        if unidecode.unidecode(word) in embedding_index:
            embedding_matrix[i] = embedding_index[unidecode.unidecode(word)]
            continue
        word = re.sub('[0-9]', '', word)
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
            continue

        unknown_words.append(word)
            
    return embedding_matrix, unknown_words