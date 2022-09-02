import cv2
import fire
import os
import pickle
import PIL

import albumentations as al
import numpy as np
import torchvision.transforms.functional as ttf

from functools import partial
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List

from PIL import Image

from ldm.modules.image_degradation import (
    degradation_fn_bsr,
    degradation_fn_bsr_light,
)


class AvidSuperRes(Dataset):

    def __init__(
        self,
        size:int=None,
        degradation:str=None,
        downscale_f:int=4,
        min_crop_f:float=0.5,
        max_crop_f:float=1.,
        names:List[str]=None,
    ):
        '''
        Avid Superresolution Dataloader

        Performs following ops in order:
        1. crops image patch with size s as a random crop
        2. resizes crop to size with cv2.area_interpolation
        3. degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: downsample factor
        :param min_crop_f: determins crop size s, s = c * min_img_side_len
            where c is sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: explained above
        '''
        super().__init__()
        self.base = self.get_base(names)
        assert size and (size / downscale_f).is_integer()
        self.size = size
        self.lr_size = int(size / downscale_f)
        assert 0 < min_crop_f <= max_crop_f <= 1
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        self.img_rescler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.pil_interpolation = False
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
                "pil_nearest": PIL.Image.NEAREST,
                "pil_bilinear": PIL.Image.BILINEAR,
                "pil_bicubic": PIL.Image.BICUBIC,
                "pil_box": PIL.Image.BOX,
                "pil_hamming": PIL.Image.HAMMING,
                "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]
            self.pil_interpolation = degradation.startswith("pil_")
            if self.pil_interpolation:
                self.degradation_process = partial(
                    ttf.resize, size=self.LR_size, interpolation=interpolation_fn)
            else:
                self.degradation_process = al.SmallestMaxSize(
                    max_size=self.lr_size, interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        example = self.base[i]
        with Image.open(example['path']) as image:
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            img = np.array(image).astype(np.uint8)
        min_side_len = min(img.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)
        self.cropper = al.RandomCrop(height=crop_side_len, width=crop_side_len)
        img = self.cropper(image=img)['image']
        img = self.img_rescler(image=img)['image']
        if self.pil_interpolation:
            image_pil = Image.fromarray(img)
            lr_image = self.degradation_process(image_pil)
            lr_image = np.array(lr_image).astype(np.uint8)
        else:
            lr_image = self.degradation_process(image=img)['image']
        return {
            'image': (img/127.5-1.0).astype(np.float32),
            'lr_image': (lr_image/127.5-1.0).astype(np.float32)
        }

    def get_base(self, names: List[str]):
        '''
        Get examples
        '''
        root_dir = Path(os.environ['AVID_ROOT_DIR'])
        examples = []
        for folder_name in names:
            img_dir = root_dir / folder_name
            flist_path = img_dir / 'flist.pkl'
            if not flist_path.exists():
                continue
            with open(flist_path, 'rb') as f:
                flist = pickle.load(f)
            for fname in flist:
                examples.append({ 'path': str(img_dir / fname) })
        return examples


class AvidSuperResTrain(AvidSuperRes):

    def __init__(self, **kwargs):
        kwargs.update({ 'names': ['Limit1'] })
        super().__init__(**kwargs)


class AvidSuperResValidation(AvidSuperRes):

    def __init__(self, num_items:int=1024, **kwargs):
        kwargs.update({ 'names': ['Random'] })
        super().__init__(**kwargs)
        self.base = self.base[:num_items]


def gen_file_list():
    avid_root_dir = Path(os.environ['AVID_ROOT_DIR'])
    for sub_dir in avid_root_dir.glob('*'):
        if not sub_dir.is_dir():
            continue
        print(f'process {sub_dir.name}')
        flist = []
        for path in tqdm(list(sub_dir.glob('*.jpg'))):
            flist.append(path.name)
        with open(sub_dir / 'flist.pkl', 'wb') as f:
            pickle.dump(flist, f)


def filter_file_list():
    avid_root_dir = Path(os.environ['AVID_ROOT_DIR'])
    for sub_dir in avid_root_dir.glob('*'):
        if not sub_dir.is_dir():
            continue
        print(f'process {sub_dir.name}')
        for path in tqdm(list(sub_dir.glob('*.jpg'))):
            with Image.open(path) as image:
                h, w = image.size
                if h * 4 < w or w * 4 < h:
                    print(f'Delete {path.name} with shape {h} x {w}')
                    os.remove(str(path))


def main(mode:str):
    if mode == 'gen':
        gen_file_list()
    elif mode == 'filter':
        filter_file_list()


if __name__ == '__main__':
    fire.Fire(main)