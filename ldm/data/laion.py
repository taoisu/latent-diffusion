import cv2
import os
import requests

import albumentations as al
import numpy as np

from datasets import load_dataset
from io import BytesIO
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        self.items = self.get_items(name)
        self.idx_map = { i: i for i in range(len(self.items)) }
        self.size = size
        os.makedirs(self.cache_dir, exist_ok=True)

    def cache_item(self, idx:int, item:Dict):
        '''
        try download and cache the image for the item, and return whether the op is successful
        '''
        url = item['URL']
        img_path = self.cache_dir / f'{idx}.jpg'
        if img_path.exists():
            return True
        else:
            headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36' }
            try:
                res = requests.get(url, headers=headers, timeout=2)
                image = Image.open(BytesIO(res.content))
                image = image.convert(mode='RGB')
            except Exception:
                return False
            image.save(img_path)
        return True

    def train_item(self, idx:int, item:Dict):
        img_name = f'{idx}.jpg'
        img_path = self.cache_dir / img_name
        if img_path.exists():
            image = Image.open(img_path)
            text = item['TEXT']
            image = image.convert('RGB')
        else:
            image = Image.new('RGB', (self.size, self.size))
            text = ""
        image = np.array(image).astype(np.uint8)
        min_side_len = min(image.shape[:2])
        self.cropper = al.RandomCrop(height=min_side_len, width=min_side_len)
        image = self.cropper(image=image)['image']
        image = self.img_rescaler(image=image)['image']
        return {
            'caption': text,
            'image': image,
        }

    def get_items(self, name:str):
        assert name in [
            'laion/laion2B-en-aesthetic',
        ]
        ds = load_dataset(name, split='train')
        return ds

    def __getitem__(self, i:int):
        idx = self.idx_map[i]
        item = self.items[idx]
        self.cache_item(idx, item)
        return self.train_item(idx, item)

    def __len__(self):
        return len(self.idx_map)

    def get_disk_items(self):
        idx_map = {}
        for i, path in enumerate(self.cache_dir.glob('*.jpg')):
            idx = int(path.stem)
            idx_map[i] = idx
        return idx_map


class LaionTextToImageTrain(LaionTextToImage):

    def __init__(self, no_download:bool=True, **kwargs):
        super().__init__(**kwargs)
        if no_download:
            self.idx_map = self.get_disk_items()


class LaionTextToImageValidation(LaionTextToImage):

    def __init__(self, no_download:bool=True, num_items:int=1024, **kwargs):
        super().__init__(**kwargs)
        if no_download:
            self.idx_map = self.get_disk_items()
        keys = list(self.idx_map.keys())[:num_items]
        self.idx_map = { k: self.idx_map[k] for k in keys}


def main():
    ds = LaionTextToImageTrain(
        name='laion/laion2B-en-aesthetic',
        size=128,
        no_download=False)
    dl = DataLoader(ds, batch_size=128, num_workers=os.cpu_count())
    for _ in tqdm(dl):
        pass


if __name__ == '__main__':
    main()