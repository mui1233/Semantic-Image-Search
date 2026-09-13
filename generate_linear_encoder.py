from __future__ import annotations

import mygrad as mg
import mynn
import numpy as np
import torch

from cogworks_data.language import get_data_path
from pathlib import Path

from classes import Image


from mygrad.nnet.initializers import glorot_normal
from mygrad.nnet.losses import margin_ranking_loss
from mynn.layers.dense import dense
from mynn.losses.mean_squared_loss import mean_squared_loss
from mynn.optimizers.sgd import SGD
from mynn.optimizers.adam import Adam
import dill as pickle

from tqdm.auto import tqdm
'''
the end goal of all of this is to create and train a model that takes a 512 dimensional (shape 512,) descriptor vector (of an image) 
and turns it into a 200 dimensional (shape 200,) embedding vector
'''

model_path = Path("model_weights.npz")

class ImageEncoder:
    def __init__(self, D_full, D_hidden):
        self.embedding_layer = dense(D_full, D_hidden, weight_initializer=glorot_normal, bias=False) 
    
    def __call__(self, x):
        return self.embedding_layer(x)
    
    @property
    def parameters(self):
        return self.embedding_layer.parameters


def load_update_image_data(data_path): # clean up the COCO dataset based on what images are not present in the Resnet-18 captioned pictures (about 100 of 80k get removed, for reference)
    with Path(data_path).open('rb') as f:
        all_data = pickle.load(f)

    with Path(get_data_path('resnet18_features.pkl')).open('rb') as f:
        resnet18_features = pickle.load(f)

    valid_images = set(resnet18_features.keys())

    
    updated_data = {
        img_id: img_object for img_id, img_object in all_data.items() if img_id in valid_images
    }

    for img_id, img_object in updated_data.items():
        img_object.add_descriptor(resnet18_features[img_id])

    return updated_data

# generates a list of tuples of length 3 holding the caption ID, the corresponding image's ID, and a random confusor image ID (guaranteed not to match the chosen image)
# dump into .pkl file because it saves like 5 minutes every time you train
# TODO: vectorize generate_data? 
def generate_data(updated_data): 
    all_image_ids = list(updated_data.keys())
    n_images = len(all_image_ids)
    
    # Generate ALL random confusor IDs at once (vectorized)
    img_indices = np.arange(n_images)
    confusor_indices = np.random.choice(n_images, size=n_images, replace=True)
    
    # Fix any cases where confusor == original (vectorized)
    mask = confusor_indices == img_indices
    while mask.any():
        confusor_indices[mask] = np.random.choice(n_images, size=mask.sum(), replace=True)
        mask = confusor_indices == img_indices
    
    # Build x_data
    x_data = []
    for i, (img_id, img_object) in enumerate(tqdm(updated_data.items(), desc="splitting data")):
        confusor_id = all_image_ids[confusor_indices[i]]
        x_data.append((img_object.caption_ids[0], img_id, confusor_id))

    x_train = np.array(x_data[:int(0.8*len(x_data))])
    x_validation = np.array(x_data[int(0.8*len(x_data)):])
    
    with open("x_train.pkl", "wb") as file:
        pickle.dump(x_train, file)
    with open("x_validation.pkl", "wb") as file:
        pickle.dump(x_validation, file)

# loads a list of all captions from the corresponding .pkl file
def load_captions_data(data_path):
    with open("all_caption.pkl", 'rb') as f:
        caption_objects: dict[int, object] = pickle.load(f)
    return caption_objects

# hyperparameters
batch_size = 32
epochs = 100
lr = 1e-4
momentum = 0.9
dim_descriptor = 512
dim_embedding = 200


# trains a linear encoder
def train_model(model, data, val_data, descriptors, captions):
    opt = Adam(model.parameters,learning_rate=lr, weight_decay=1e-5)

    # all_image_ids = list(data.keys())

    improvement_limit = 10
    improvement_counter = 0

    delta = 0.0001
    best_val_accuracy = 0

    for epoch_cnt in range(epochs):
        idxs = np.arange(len(data))
        np.random.shuffle(idxs)

        batch_accuracies = []
        
        for batch_cnt in tqdm(range(0, len(data) // batch_size), desc="running batch"):
            batch_indices = idxs[(batch_cnt * batch_size):((batch_cnt + 1) * batch_size)]
            batch_triplets = data[batch_indices]

            caption_embeddings = np.array([captions[triplet[0]].embedding for triplet in batch_triplets]) # a bunch of 200 dimensional embeddings
            batch_true = np.array([descriptors[triplet[1]].descriptor for triplet in batch_triplets]).squeeze() # a bunch of 512 dimensional descriptor vectors
            batch_confusor = np.array([descriptors[triplet[2]].descriptor for triplet in batch_triplets]).squeeze()
        

            # caption_embeddings = mg.Tensor([captions[triplet[0]].embedding for triplet in data[batch_indices]])
            # batch_true = mg.Tensor([descriptors[triplet[1]].descriptor for triplet in data[batch_indices]])
            # batch_confusor = mg.Tensor([descriptors[triplet[2]].descriptor for triplet in data[batch_indices]])
            
            pred_true = model(batch_true)
            pred_confusor = model(batch_confusor)
            
            dot_true = mg.sum(pred_true * caption_embeddings, axis=1)
            dot_confusor = mg.sum(pred_confusor * caption_embeddings, axis=1)

            loss = margin_ranking_loss(dot_true, dot_confusor, y=1, margin=0.25)

            loss.backward()
            opt.step()
            accuracy = np.mean(dot_true.data > dot_confusor.data)

            batch_accuracies.append(accuracy)
        val_accuracy = validate_model(model, val_data, captions, descriptors)
        print(f"epoch {epoch_cnt + 1}/{epochs} | train acc: {np.mean(batch_accuracies):.3f} | val acc: {val_accuracy:.3f}")

        if val_accuracy > best_val_accuracy + delta:
            best_val_accuracy = val_accuracy
            improvement_counter = 0
            print(f"new best model, saving to {model_path}")
            save_weights(model, model_path)
        else:
            improvement_counter += 1
        
        if improvement_counter >= improvement_limit:
            print(f"no improvement for {improvement_counter} epochs, aborting")
            break
    return model

# probably move a bunch of stuff to main? or not, because this doesn't have to happen in main
# only when training the model

def save_weights(model, path):
    params = [param.data for param in model.parameters]
    np.savez(path, *params)
    
def load_weights(model, path):
    loaded_weights = np.load(path)
    for i, param in enumerate(model.parameters):
        param.data[...] = loaded_weights[f"arr_{i}"]


def validate_model(model, data, captions, descriptors):
    val_accuracies = []

    with mg.no_autodiff:
        for batch_cnt in tqdm(range(0, len(data) // batch_size), desc="testing model"): 
            batch_indices = np.arange(len(data))
            batch_indices = batch_indices[batch_cnt * batch_size:batch_cnt * batch_size + batch_size]
            batch_triplets = data[batch_indices]

            caption_embeddings = np.array([captions[triplet[0]].embedding for triplet in batch_triplets]) # a bunch of 200 dimensional embeddings
            batch_true = np.array([descriptors[triplet[1]].descriptor for triplet in batch_triplets]).squeeze() # a bunch of 512 dimensional descriptor vectors
            batch_confusor = np.array([descriptors[triplet[2]].descriptor for triplet in batch_triplets]).squeeze()

            pred_true = model(batch_true)
            pred_confusor = model(batch_confusor)

            dot_true = mg.sum(pred_true * caption_embeddings, axis=1)
            dot_confusor = mg.sum(pred_confusor * caption_embeddings, axis=1)

            accuracy = np.mean(dot_true.data > dot_confusor.data)
            val_accuracies.append(accuracy)
    return np.mean(val_accuracies)


def train_and_validate():
    attempt_load_weights = False

    descriptors = load_update_image_data('all_image.pkl')
    captions = load_captions_data('all_caption.pkl')

    train_data = Path("x_train.pkl")
    val_data = Path("x_validation.pkl")

    model = ImageEncoder(dim_descriptor, dim_embedding)
    
    if not train_data.exists() or not val_data.exists():
        print("generating data (training data not initialized)")
        generate_data(descriptors) 
    else:
        print("loaded data")

    with train_data.open('rb') as f:
        x_train = pickle.load(f)
    with val_data.open('rb') as f:
        x_val = pickle.load(f)

    if attempt_load_weights:
        if model_path.exists():
            load_weights(model, model_path)
            print(f"loading model from {model_path}")
        else: 
            print(f"model not found at {model_path}")

    model = train_model(model, x_train, x_val, descriptors, captions)

    # test model accuracy on validation set
    test_accuracy = validate_model(model, x_val, captions, descriptors)
    print(f"final accuracy: {np.mean(test_accuracy)}") # 0-1, a percentage

    save_weights(model, model_path)
    print(f"saved model to {model_path}")

    embed_all_images(model, descriptors)

# utility function to save and load model weights

def embed_all_images(model, descriptors : dict[int, Image]):
    # Extract all descriptors and IDs in order
    img_ids = list(descriptors.keys())
    img_descriptors = np.array([descriptors[img_id].descriptor for img_id in img_ids])
    
    # Process ALL images at once (batched)
    with mg.no_autodiff: 
        embeddings = model(img_descriptors)
    
    # Normalize all embeddings at once
    embeddings = embeddings / np.sqrt(np.sum(embeddings**2, axis=1, keepdims=True))
    
    # Assign back to individual images
    for i, img_id in enumerate(tqdm(img_ids, desc="embedding images")):
        descriptors[img_id].add_embedding(embeddings[i])
    
    with open("all_image.pkl", "wb") as file:
        pickle.dump(descriptors, file, protocol=-1)

'''
Create model to learn W_embed such that d_img * W_embed = w_embed
d_img = image descriptor
W_embed = learnable weights
w_embed = image caption

def calculate_loss():

def calculate_accuracy():

def train_model():

def validation_accuracy():

'''