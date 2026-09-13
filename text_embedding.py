import pickle
import numpy as np
from collections import Counter
import re
from gensim.models import KeyedVectors
from cogworks_data.language import get_data_path

import tqdm

def load_captions(pkl_path): 
    """Load dict of captions from a pickle file."""
    with open(pkl_path, 'rb') as f:
        captions: dict[int, object] = pickle.load(f)
    return {id: caption.caption_text for id, caption in captions.items()} # create of dictionary of id and text files

def process_text(caption):
    """Preprocess the caption text by removing punctuation and converting to lowercase
    Returns a list of tokens."""
    caption = caption.lower()
    caption = re.sub(r"[^\w\s]", "", caption)
    return caption.split() # Turn into tokens: "Cats are cool" -> ["cats", "are", "cool"]

def idf_text(captions): 
    """
    Input: captions, List[str]
    Output: dict[str, int]
    """
    # "I am happy" "I am sad" -> {"i": 2, "am": 2, "happy": 1, "sad": 1}
    N = len(captions) # 
    df:dict[str,int] = {}
    for caption in captions: # Each caption in captions
        tokens = process_text(caption) # Splits up into a list of words
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1 #  This just appends the dictionary by 0 if the word t doesn't exist yet. 
    return {token: np.log10(N/df_t) for token, df_t in df.items()} # 

def load_glove():
    """Load the GloVe word vectors from a file."""
    filename = "glove.6B.200d.txt.w2v"
    # this takes a while to load -- keep this in mind when designing your capstone project
    glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)
    return glove

def embed_text(caption, glove, idf):
    """
    Embed a caption using GLoVe and IDF.
    Input: caption:str, glove: KeyedVectors, idf: dict[str, float]
    Output: np.ndarray, shape (200,)
    """
    w_text = np.zeros(glove.vector_size, dtype=np.float32) # Initialize a zero vector of the same size as GloVe vectors
    tokens = process_text(caption)
    for t in tokens: 
        if t in glove: 
            weight = glove[t] # shape (200,)
            w_text += idf.get(t, 0) * weight # float * shape(200,) = shape(200,)
    norm = np.linalg.norm(w_text)
    if norm > 0:
        return w_text/norm
    else:
        return w_text