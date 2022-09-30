import cv2
import fire
import os
import pickle
import requests

import albumentations as al
import numpy as np
import torchvision.transforms.functional as ttf

from datasets import load_dataset
from functools import partial
from io import BytesIO
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, Tuple

from PIL import Image, ImageFile

from ldm.modules.image_degradation import (
    degradation_fn_bsr,
    degradation_fn_bsr_light,
)

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
        assert img_path.exists()
        try:
            with Image.open(img_path) as image:
                image = image.convert('RGB')
                img = np.array(image).astype(np.uint8)
        except Exception as e:
            print(e)
            sz = self.size
            img = np.ones((sz, sz, 3), dtype=np.uint8)*255
        text = item['TEXT'] or ''
        min_side_len = min(img.shape[:2])
        self.cropper = al.RandomCrop(height=min_side_len, width=min_side_len)
        img = self.cropper(image=img)['image']
        img = self.img_rescaler(image=img)['image']
        img = (img/127.5-1.0).astype(np.float32)
        if self.dropout > 0 and np.random.random() < self.dropout:
            text = ''
        return {
            'caption': text,
            'image': img,
        }

    def load_items(self, name:str):
        assert name in [
            'laion/laion2B-en-aesthetic',
        ]
        ds = load_dataset(name, split='train')
        ds = ds.remove_columns([col for col in ds.column_names if col != 'TEXT'])
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


class LaionSuperRes(LaionTextToImage):

    def __init__(
        self,
        name:str,
        size:int,
        lr_size:int,
        dropout:float=0.0,
        idx_map_name:str=None,
        num_items:int=None,
        degradation:str=None,
    ):
        '''
        Laion Super Resolution Dataset

        Performs the following ops:
        1. center crops the image with size of min side len
        2. resize the crop to the size
        '''
        super().__init__(
            name=name,
            size=size,
            dropout=dropout,
            idx_map_name=idx_map_name,
        )
        self.lr_size = lr_size
        if num_items is not None:
            vals = sorted(list(self.idx_map.values()))
            vals = vals[:num_items]
            self.idx_map = { k: vals[k] for k in range(len(vals)) }
        self.pil_interpolation = False
        downscale_f = size // lr_size
        if degradation == 'bsrgan':
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)
        elif degradation == 'bsrgan_light':
            self.degradation_process = partial(degradation_fn_bsr_light)
        else:
            interpolation_fn = {
                "cv_nearest": cv2.INTER_NEAREST,
                "cv_bilinear": cv2.INTER_LINEAR,
                "cv_bicubic": cv2.INTER_CUBIC,
                "cv_area": cv2.INTER_AREA,
                "cv_lanczos": cv2.INTER_LANCZOS4,
                "pil_nearest": Image.NEAREST,
                "pil_bilinear": Image.BILINEAR,
                "pil_bicubic": Image.BICUBIC,
                "pil_box": Image.BOX,
                "pil_hamming": Image.HAMMING,
                "pil_lanczos": Image.LANCZOS,
            }[degradation]
            self.pil_interpolation = degradation.startswith("pil_")
            if self.pil_interpolation:
                self.degradation_process = partial(
                    ttf.resize, size=self.lr_size, interpolation=interpolation_fn)
            else:
                self.degradation_process = al.SmallestMaxSize(
                    max_size=self.lr_size, interpolation=interpolation_fn)

    def prep_lr_image(self, image:np.ndarray):
        image = ((image+1.0)*127.5).astype(np.uint8)
        lr_image = self.degradation_process(image)['image']
        lr_image = (lr_image/127.5-1.0).astype(np.float32)
        return lr_image

    def __getitem__(self, i:int):
        item = super().__getitem__(i)
        item['lr_image'] = self.prep_lr_image(item['image'])
        return item


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
            with Image.open(BytesIO(res.content)) as image:
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