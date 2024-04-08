from torch.utils.data import Dataset
from PIL import Image
import logging
import json
import re
import os


class NewsImageCaptionDataset(Dataset):
    """
    News Image Caption Dataset from Good News datasource

    Note: Here we are loading all the images in memory. This is not a good practice. We should load images on the fly
    when required.
    """

    def __init__(self, caption_file, image_dir, image_transform=None, caption_transform=None, article_transform=None, caption_vocab=None):
        self.caption_transform = caption_transform
        self.article_transform = article_transform
        self.image_transform = image_transform
        self.caption_vocab = caption_vocab
        self.captions = []
        self.articles = []
        self.data = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading Data from {caption_file}")

        json_data = json.load(open(caption_file))
        for key, item in json_data.items():
            article = item['article']
            for idx, caption in item['images'].items():
                # filter out captions with no words
                if not self._is_valid_caption(caption):
                    self.logger.debug(fr"Invalid caption for {key}_{idx}: {repr(caption)}")
                    continue

                image_name = f"{key}_{idx}.jpg"
                if not self._is_image_available(f"{image_dir}/{image_name}"):
                    self.logger.debug(f"Image not found: {image_name}")
                    continue
                
                self.captions.append(caption)
                self.articles.append(article)
                
                # TODO: make this step in getitem part of the pipeline
                # img = self._load_image(f"{image_dir}/{image_name}")
                self.data.append({"image": f"{image_dir}/{image_name}", "caption": caption, "article": article})

        self.logger.info(f"Loaded {len(self.data)} images")

    @staticmethod
    def _is_image_available(image_path):
        return os.path.exists(image_path)

    @staticmethod
    def _load_image(image_path):
        img = Image.open(image_path)
        return img

    @staticmethod
    def _is_valid_caption(caption):
        return re.search(r'\w+', caption) is not None

    def __len__(self):
        return len(self.data)
    
    def get_captions(self):
        return self.captions
    
    def get_articles(self):
        return self.articles
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img = self._load_image(item["image"])
        return {"image": self.image_transform(img) if self.image_transform else img,
                "caption": self.caption_transform(self.caption_vocab, item["caption"]) if self.caption_transform else item["caption"],
                "article": self.article_transform(item["article"]) if self.article_transform else item["article"]}