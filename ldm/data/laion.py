import cv2
import fire
import os
import pickle
import requests

import albumentations as al
import numpy as np

from datasets import load_dataset
from io import BytesIO
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LaionTextToImage(Dataset):

    def __init__(
        self,
        name:str,
        size:int,
        dropout:float=0.0,
        idx_map_name:str=None,
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
        self.items = self.load_items(name)
        if idx_map_name is None:
            self.idx_map = { i: i for i in range(len(self.items)) }
        else:
            self.idx_map = self.get_disk_items(idx_map_name)
        self.size = size
        self.dropout = dropout
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
        assert img_path.exists()
        image = Image.open(img_path)
        text = item['TEXT'] or ''
        image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)
        min_side_len = min(image.shape[:2])
        self.cropper = al.RandomCrop(height=min_side_len, width=min_side_len)
        image = self.cropper(image=image)['image']
        image = self.img_rescaler(image=image)['image']
        image = (image/127.5-1.0).astype(np.float32)
        if self.dropout > 0 and np.random.random() < self.dropout:
            text = ''
        return {
            'caption': text,
            'image': image,
        }

    def load_items(self, name:str):
        assert name in [
            'laion/laion2B-en-aesthetic',
        ]
        ds = load_dataset(name, split='train')
        return ds

    def __getitem__(self, i:int):
        idx = self.idx_map[i]
        item = self.items[idx]
        return self.train_item(idx, item)

    def __len__(self):
        return len(self.idx_map)

    def get_disk_items(self, idx_map_name:str):
        idx_map_path = self.cache_dir / f'{idx_map_name}.pkl'
        with open(idx_map_path, 'rb') as f:
            idx_map = pickle.load(f)
        return idx_map


class LaionTextToImageTrain(LaionTextToImage):

    def __init__(self,
        **kwargs):
        super().__init__(**kwargs)


class LaionTextToImageValidation(LaionTextToImage):

    def __init__(self,
        num_items:int=1024,
        **kwargs):
        super().__init__(**kwargs)
        vals = sorted(list(self.idx_map.values()))
        vals = vals[:num_items]
        self.idx_map = { k: vals[k] for k in range(len(vals)) }


def try_download(obj:Tuple):
    item, idx, cache_dir = obj
    url = item['URL']
    img_path = cache_dir / f'{idx}.jpg'
    if img_path.exists():
        return
    else:
        headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36' }
        try:
            res = requests.get(url, headers=headers, timeout=2)
            image = Image.open(BytesIO(res.content))
            image = image.convert(mode='RGB')
            image.save(img_path)
        except Exception:
            return


def download_dataset(name:str):
    assert name in [
        'laion/laion2B-en-aesthetic',
    ]
    dataset = load_dataset(name, split='train')
    num_items = len(dataset)
    cache_dir = Path(os.environ['LAION_ROOT_DIR']) / name
    pool = Pool(processes=os.cpu_count())
    tasks = zip(dataset, range(0, num_items), repeat(cache_dir, num_items))
    for _ in tqdm(pool.imap_unordered(try_download, tasks), total=num_items):
        pass


def gen_idx_map(name:str):
    assert name in [
        'laion/laion2B-en-aesthetic',
    ]
    cache_dir = Path(os.environ['LAION_ROOT_DIR']) / name
    idx_map = {}
    for i, jpg_path in tqdm(enumerate(cache_dir.glob('*.jpg'))):
        idx_map[i] = int(jpg_path.stem)
    num_imgs = len(idx_map)
    idx_map_name = f'idx_map_{num_imgs}.pkl'
    with open(cache_dir / idx_map_name, 'wb') as f:
        pickle.dump(idx_map, f)
    print(idx_map_name)


def main(
    mode:str,
    name:str,
):
    if mode == 'download':
        download_dataset(name)
    elif mode == 'idx_map':
        gen_idx_map(name)


if __name__ == '__main__':
    fire.Fire(main)