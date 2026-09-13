class Image:
    def __init__(self, id, coco_url):
        self.id = id
        self.coco_url = coco_url
        
        self.caption_ids = []
        self.caption_texts = []
        
        self.descriptor = None
        self.W = None
        self.embedding = None

    def add_caption(self, cap_class):
        self.caption_ids.append(cap_class.get_id())
        self.caption_texts.append(cap_class.get_text())
    
    def add_descriptor(self, descriptor):
        self.descriptor = descriptor
    
    def add_embedding(self, embedding):
        self.embedding = embedding
    
class Caption:
    def __init__(self, id, image_id, caption_text):
        self.id = id
        self.image_id = image_id
        self.caption_text = caption_text
        self.embedding = None

    def get_id(self):
        return self.id


    def get_image_id(self):
        return self.image_id


    def get_text(self):
        return self.caption_text
    
    def update_embedding(self, embedding):
        self.embedding = embedding