import cv2
import os
import requests

import albumentations as al
import numpy as np

from datasets import load_dataset
from io import BytesIO
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
        self.size = size
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
        img_name = Path(url).name.split('?')[0]
        img_path = self.cache_dir / img_name
        text = item['TEXT']
        if img_path.exists():
            image = Image.open(img_path)
        else:
            headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36' }
            res = requests.get(url, headers=headers)
            try:
                image = Image.open(BytesIO(res.content))
            except:
                image = Image.new('RGB', (self.size, self.size))
                text = ""
                img_path = img_path.parent / '_blank.jpg'
            image.save(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)
        min_side_len = min(image.shape[:2])
        self.cropper = al.RandomCrop(height=min_side_len, width=min_side_len)
        image = self.cropper(image=image)['image']
        image = self.img_rescaler(image=image)['image']
        example = {
            'caption': text,
            'image': image,
        }
        return example


class LaionTextToImageTrain(LaionTextToImage):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LaionTextToImageValidation(LaionTextToImage):

    def __init__(self, ratio:float, **kwargs):
        super().__init__(**kwargs)
        self.base = self.base.train_test_split(test_size=ratio, shuffle=True)['test']