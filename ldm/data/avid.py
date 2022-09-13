import cv2
import fire
import json
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


class AvidInpaint(Dataset):

    def __init__(
        self,
        size:int=None,
        min_pad_f:float=20,
        max_pad_f:float=60,
        limit:int=40,
        ratio:int=10,
        names:List[str]=None,
    ):
        '''
        Avid Inpaint Dataset

        Performs following ops in order:
        1. open ocr.json, randomly pick one line to be inpainted
        2. crop the surrounding region of the line
        3. generate masked crop
        '''
        super().__init__()
        self.base = self.get_base(names)
        self.size = size
        self.limit = limit
        self.ratio = ratio
        assert 0 < min_pad_f <= max_pad_f
        self.min_pad_f = min_pad_f
        self.max_pad_f = max_pad_f
        self.img_rescler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        example = self.base[i]
        try:
            with open(example['ocr_path'], 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            words = ocr_data['analyzeResult']['pages'][0]['words']
        except Exception as e:
            print(e)
            words = []
        if len(words) == 0:
            ret = {
                'image': np.ones((self.size, self.size, 3), dtype=np.uint8)*255,
                'mask': np.zeros((self.size, self.size, 1), dtype=np.float32),
                'text': '',
            }
        else:
            j = np.random.randint(0, len(words))
            line = self.merge_words(words, j)
            text, bbox = line['content'], line['polygon']
            xmin, xmax = min(bbox[0::2]), max(bbox[0::2])
            ymin, ymax = min(bbox[1::2]), max(bbox[1::2])
            x_span, y_span = xmax - xmin, ymax - ymin
            pad_span = int((self.min_pad_f + np.random.rand() * (self.max_pad_f - self.min_pad_f)))
            crop_size = x_span + pad_span
            l = min(int(pad_span * np.random.rand()), xmin)
            t = min(int(max(crop_size - y_span, 0) * np.random.rand()), ymin)
            x_start, y_start = xmin - l, ymin - t
            try:
                with Image.open(example['img_path']) as image:
                    if not image.mode == 'RGB':
                        image = image.convert('RGB')
                    img = np.array(image).astype(np.uint8)
                crop = img[y_start:y_start+crop_size, x_start:x_start+crop_size, :]
                h, w = crop.shape[:2]
                crop_size = max(h, w)
                crop = np.pad(
                    crop,
                    [(0,crop_size-h),(0,crop_size-w),(0,0)],
                    mode='constant',
                    constant_values=255)
                mask = np.zeros((crop_size, crop_size, 1), dtype=np.uint8)
                mask[t:t+ymax-ymin, l:l+xmax-xmin] = 255
                out = self.img_rescler(image=crop, mask=mask)
                crop, mask = out['image'], (out['mask'] != 0).astype(np.float32)
            except Exception as e:
                print(e)
                crop = np.ones((self.size, self.size, 3), dtype=np.uint8)*255
                mask = np.zeros((self.size, self.size, 1), dtype=np.float32)
                mask[:8, :8] = 1
                text = ''
            ret = {
                'image': crop,
                'mask': mask,
                'text': text,
            }
        ret['image'] = (ret['image']/127.5-1.0).astype(np.float32)
        return ret

    def merge_bbox(self, bbox_l: List, bbox_r: List):
        xrelax, yrelax = 2, 2
        xmin = min(bbox_l[0::2] + bbox_r[0::2])
        xmax = max(bbox_r[0::2] + bbox_r[0::2])
        ymin = min(bbox_l[1::2] + bbox_r[1::2])
        ymax = max(bbox_r[1::2] + bbox_r[1::2])
        return [
            xmin - xrelax, ymin - yrelax,
            xmax + xrelax, ymin - yrelax,
            xmax + xrelax, ymax + yrelax,
            xmin - xrelax, ymax + yrelax]

    def merge_words(self, words: List, j: int):
        line = words[j]
        bbox = line['polygon']
        ymin, ymax = min(bbox[1::2]), max(bbox[1::2])
        while len(line['content']) < self.limit:
            if j + 1 < len(words):
                bbox_next = words[j + 1]['polygon']
                bbox_merge = self.merge_bbox(bbox, bbox_next)
                xspan_merge, yspan_merge = bbox_merge[2] - bbox_merge[0], bbox_merge[-1] - bbox_merge[1]
                y_avg = 0.5 * (min(bbox_next[1::2])+max(bbox_next[1::2]))
                if ymin <= y_avg <= ymax and xspan_merge <= self.ratio * yspan_merge:
                    line['content'] += f' {words[j + 1]["content"]}'
                    line['polygon'] = bbox_merge
                else:
                    break
                j += 1
            else:
                break
        return line

    def get_base(self, names: List[str]):
        '''
        Get examples
        '''
        root_dir = Path(os.environ['AVID_ROOT_DIR'])
        examples = []
        for folder_name in names:
            img_dir = root_dir / folder_name
            flist_path = img_dir / 'flist_ocr.pkl'
            if not flist_path.exists():
                continue
            with open(flist_path, 'rb') as f:
                flist = pickle.load(f)
            for ocr_name in flist:
                img_name = ocr_name[:-len('.ocr.json')]
                examples.append({
                    'ocr_path': str(img_dir / ocr_name),
                    'img_path': str(img_dir / img_name),
                })
        return examples


class AvidInpaintTrain(AvidInpaint):

    def __init__(self, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Limit1'] })
        super().__init__(**kwargs)

class AvidInpaintValidation(AvidInpaint):

    def __init__(self, num_items:int=1024, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Random'] })
        super().__init__(**kwargs)
        self.base = self.base[:num_items]


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
        Avid Superresolution Dataset

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
        try:
            img = self.img_rescler(image=img)['image']
        except Exception as e:
            print(e)
            img = np.ones((self.size, self.size, 3), dtype=np.uint8)*255
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
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Limit1'] })
        super().__init__(**kwargs)


class AvidSuperResValidation(AvidSuperRes):

    def __init__(self, num_items:int=1024, **kwargs):
        if 'names' not in kwargs:
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


def gen_ocr_list():
    avid_root_dir = Path(os.environ['AVID_ROOT_DIR'])
    for sub_dir in avid_root_dir.glob('*'):
        if not sub_dir.is_dir():
            continue
        print(f'process {sub_dir.name}')
        flist = []
        num_miss = 0
        for ocr_path in tqdm(list(sub_dir.glob('*.ocr.json'))):
            img_path = ocr_path.parent / ocr_path.name[:-len('.ocr.json')]
            if img_path.exists():
                flist.append(ocr_path.name)
            else:
                num_miss += 1
        with open(sub_dir / 'flist_ocr.pkl', 'wb') as f:
            pickle.dump(flist, f)
        print(f'total: {len(flist)} ocr files; miss: {num_miss}')


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


def test_inpaint_dataset():
    ds = AvidInpaint(128, names=['Limit100'])
    for i in range(len(ds)):
        example = ds[i]
        print(example['text'])
        image = ((example['image']+1)*127.5).astype(np.uint8)
        Image.fromarray(image).save('a.jpg')
        mask = ((example['mask'])*255).astype(np.uint8)
        mask = np.repeat(mask, 3, -1)
        Image.fromarray(mask).save('b.jpg')


def main(mode:str):
    if mode == 'gen':
        gen_file_list()
    elif mode == 'filter':
        filter_file_list()
    elif mode == 'gen_ocr':
        gen_ocr_list()
    elif mode == 'test_inpaint':
        test_inpaint_dataset()


if __name__ == '__main__':
    fire.Fire(main)