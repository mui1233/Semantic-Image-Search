from cogworks_data.language import get_data_path
from pathlib import Path
import json
import pickle

from tqdm.auto import tqdm

from classes import Image, Caption

def load_and_embed():
    image_path = "all_image.pkl"
    caption_path = "all_caption.pkl"
    # Load COCO metadata
    filename = get_data_path("captions_train2014.json")
    with Path(filename).open() as f:
        coco_data = json.load(f)

        def get_id(self):
            return self.id


        def get_image_id(self):
            return self.image_id


        def get_text(self):
            return self.caption_text
        

    all_images = {} # img_id: img_class
    all_captions = {} # caption_id: caption_class

    for img in tqdm(coco_data['images'], desc="generating all_image.pkl"):
        img_class = Image(img['id'], img['coco_url'])
        all_images[img['id']] = img_class

    for caption in tqdm(coco_data['annotations'], desc="generating all_caption.pkl"):
        cap_class = Caption(caption['id'], caption['image_id'], caption['caption'])
        all_captions[caption['id']] = cap_class
        corr_img = all_images[caption['image_id']]
        corr_img.add_caption(cap_class)

    
    with open(image_path, "wb") as file:
        pickle.dump(all_images, file)

    
    with open(caption_path, "wb") as file:
        pickle.dump(all_captions, file)