from torch.utils.data import Dataset
from PIL import Image
import json
import re


class NewsImageCaptionDataset(Dataset):
    """
    News Image Caption Dataset from Good News datasource

    Note: Here we are loading all the images in memory. This is not a good practice. We should load images on the fly
    when required.
    """

    def __init__(self, caption_file, image_dir, image_transform=None, caption_transform=None, article_transform=None):
        json_data = json.load(open(caption_file))
        self.data = []

        for key, item in json_data.items():
            article = item['article']
            for idx, caption in item['images'].items():
                # filter out captions with no words
                if not self._is_valid_caption(caption):
                    continue

                # TODO: make this step in getitem part of the pipeline
                img = self._load_image(f"{image_dir}/{key}_{idx}.jpg")
                self.data.append({"image": img, "caption": caption, "article": article})

    @staticmethod
    def _load_image(image_path):
        img = Image.open(image_path)
        return img

    @staticmethod
    def _is_valid_caption(caption):
        return re.search(r'\w+', caption) is not None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
