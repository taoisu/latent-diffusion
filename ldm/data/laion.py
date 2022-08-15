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

    def train_item(self, idx:int, item:Dict):
        img_name = f'{idx}.jpg'
        img_path = self.cache_dir / img_name
        image = Image.open(img_path)
        text = item['TEXT']
        image = image.convert('RGB')
        image = np.array(image).astype(np.uint8)
        min_side_len = min(image.shape[:2])
        self.cropper = al.RandomCrop(height=min_side_len, width=min_side_len)
        image = self.cropper(image=image)['image']
        image = self.img_rescaler(image=image)['image']
        image = (image/127.5-1.0).astype(np.float32)
        if self.dropout > 0 and np.random.random() < self.dropout:
            text = ""
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
        keys = list(self.idx_map.keys())[:num_items]
        self.idx_map = { k: self.idx_map[k] for k in keys}


def try_download(item:Tuple[Dataset, int, str]):
    ds, i, name = item
    cache_dir = Path(os.environ['LAION_ROOT_DIR']) / name
    item = ds[i]
    url = item['URL']
    img_path = cache_dir / f'{i}.jpg'
    if img_path.exists():
        return
    else:
        try:
            headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36' }
            res = requests.get(url, headers=headers, timeout=2)
            image = Image.open(BytesIO(res.content))
            image = image.convert(mode='RGB')
            image.save(img_path)
        except Exception:
            return


def prepare_dataset(name:str):
    ds = load_dataset(name, split='train')
    num_items = len(ds)
    pool = Pool(processes=os.cpu_count())
    tasks = zip(repeat(ds, num_items), range(0, num_items), repeat(name, num_items))
    for _ in tqdm(pool.imap_unordered(try_download, tasks), total=num_items):
        pass


def prepare_map(name:str):
    cache_dir = Path(os.environ['LAION_ROOT_DIR']) / name
    paths = list(cache_dir.glob('*.jpg'))
    idx_map = {}
    for i, path in tqdm(enumerate(paths)):
        idx_map[i] = int(path.stem)
    with open(cache_dir / f'idx_map_{len(paths)}.pkl', 'wb') as f:
        pickle.dump(idx_map, f)
    print(f'dump complete: idx_map_{len(paths)}.pkl')


def main(mode:str, name:str='laion/laion2B-en-aesthetic'):
    if mode == 'dataset':
        prepare_dataset(name)
    elif mode == 'map':
        prepare_map(name)


if __name__ == '__main__':
    fire.Fire(main)