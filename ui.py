import streamlit as st
import pickle
import numpy as np
from gensim.models import KeyedVectors
from classes import Image, Caption 
from text_embedding import load_captions, embed_text, load_glove, idf_text, process_text

import mygrad as mg

# how to run: 
# in terminal, type streamlit run ui.py
# a local server should pop up in your browser

st.title("Trouble-Shooters")

query = st.text_input("Enter your search query:", placeholder="e.g. horses on a beach")
print("loading glove")
glove = KeyedVectors.load_word2vec_format("glove.6B.200d.txt", binary=False)

print("glove loaded")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a non-empty query.")
    else:
        st.write(f"Searching for: **{query}**")
        print(f"query: {query}")
        
        
        # TODO: embeddings, querying
        k=4 # ADD QUERYING PORTION LTR
        # Preprocessing text
        processed_cap = process_text(query)
        
        # Turning query into W
        with open('idf.pkl', 'rb') as f: 
            idf = pickle.load(f)

        
        # glove = load_glove() 
        # glove.save("glove.200d.kv")
        
        

        W_norm = embed_text(query, glove, idf) # shape (200,) our query normalized

        # Compare against images
        with open("all_image.pkl", "rb") as f:
            images = pickle.load(f)

        # Create W_imgs
        W_imgs = np.vstack([image.embedding.data for image in images.values() if image.embedding is not None])

        print(f"W_imgs shape: {W_imgs.shape}")  # Debugging line
        
        scores = np.dot(W_imgs, W_norm) # Similarity Score
        top_imgs = np.argsort(scores)[-k:][::-1]  # Get top k indices inside an array
        #create list of only images that have embeddings
        valid_images = [img for img in images.values() if img.embedding is not None]
        #loop through top matching image indices
        for i, idx in enumerate(top_imgs):
            #display result number and similarty score
            st.write(f"**Result {i+1}.** {scores[idx]:.3f} similarity")
            #display the image
            st.image(valid_images[idx].coco_url, width=300)