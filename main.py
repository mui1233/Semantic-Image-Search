import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle
import numpy as np
from gensim.models import KeyedVectors
from text_embedding import load_captions, process_text, idf_text, load_glove, embed_text
from parse_dataset import load_and_embed
from generate_linear_encoder import train_and_validate
import time
def main():
    # after this, all_image.pkl holds a bunch of Image objects, each holding a bunch of caption objects, which then in turn hold their own embedding
    load_and_embed()
    
    # Load caption (text only) from pickle file
    captions = load_captions("all_caption.pkl")
    # Process the captions to compute IDF values
    idf = idf_text(list(captions.values()))
    with open('idf.pkl', 'wb') as f:
        pickle.dump(idf, f)
    
    # Load GloVe word vectors
    # glove = load_glove()
    glove = KeyedVectors.load_word2vec_format("glove.6B.200d.txt", binary=False)

    # Load caption objects from a pickle file
    with open("all_caption.pkl", 'rb') as f:
        caption_objects: dict[int, object] = pickle.load(f)
        
    # place each embedding into each caption
    # i.e. every caption is now responsible for its own embedding
    for id, caption in caption_objects.items():
        embedding = embed_text(caption.get_text(), glove, idf)
        caption.update_embedding(embedding)
    
    # store caption_objects to be used in model training
    with open("all_caption.pkl", "wb") as file:
        pickle.dump(caption_objects, file)
        

    # train and validate model
    train_and_validate()

main()