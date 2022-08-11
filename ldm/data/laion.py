import cv2
import os
import requests

import albumentations as al
import numpy as np

from datasets import load_dataset
from pathlib import Path
from torch.utils.data import Dataset

from PIL import Image


class LaionTextToImage(Dataset):

    def __init__(
        self,
        name:str,
        size:int,
    ):
        '''
        Laion TextToImage Dataset

        Performs follow ops:
        1. download the image if it's not in cache
        2. center crops the image with size of min side len
        3. resizes crop to the size
        '''
        super().__init__()
        self.img_rescaler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.cache_dir = Path(os.environ['LAION_ROOT_DIR']) / name
        self.base = self.get_base(name)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_base(self, name:str):
        assert name in [
            'laion/laion2B-en-aesthetic',
        ]
        return load_dataset(name, split='train')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        item = self.base[i]
        url= item['URL']
        img_name = Path(url).name
        img_path = self.cache_dir / img_name
        with open(img_path, 'wb') as f:
            res = requests.get(url)
            f.write(res.content)
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)
        min_side_len = min(image.shape[:2])
        self.cropper = al.CenterCrop(height=min_side_len, width=min_side_len)
        image = self.cropper(image=image)['image']
        image = self.img_rescaler(image=image)['image']
        example = {
            'caption': item['TEXT'],
            'image': image,
        }
        return example


class LaionTextToImageTrain(LaionTextToImage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LaionTextToImageValidation(LaionTextToImage):

    def __init__(self, subset_size:int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.base = self.base[:subset_size]