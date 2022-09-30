import cv2
import fire
import json
import os
import pickle
import PIL
import shutil

import albumentations as al
import numpy as np
import torchvision.transforms.functional as ttf

from functools import partial
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List

from PIL import Image, ImageDraw, ImageFont

from ldm.modules.image_degradation import (
    degradation_fn_bsr,
    degradation_fn_bsr_light,
)


class BizcardInpaint(Dataset):

    def __init__(
        self,
        size:int,
        min_pad_f:float=20,
        max_pad_f:float=60,
        limit:int=40,
        ratio:int=10,
        names:List[str]=None,
    ):
        '''
        Bizcard Inpaint Dataset

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
        # self.img_rescler = al.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i:int):
        example = self.base[i]
        try:
            with Image.open(example['img_path']) as image:
                if not image.mode == 'RGB':
                    image = image.convert('RGB')
                img = np.array(image).astype(np.uint8)
            image_data = (img/127.5-1.0).astype(np.float32)
        except Exception as e:
            print(e)

        ret = {
                'image': image_data,
            }
        return ret

    def get_base(self, names: List[str]):
        '''
        Get examples
        '''
        root_dir = Path(os.environ['BIZCARD_ROOT_DIR'])
        examples = []
        for folder_name in names:
            img_dir = root_dir / folder_name
            flist_path = img_dir / 'flist.pkl'
            if not flist_path.exists():
                continue
            with open(flist_path, 'rb') as f:
                flist = pickle.load(f)
            for img_name in flist:
                examples.append({
                    # 'ocr_path': str(img_dir / ocr_name),
                    'img_path': str(img_dir / img_name),
                })
        return examples

class BizcardSuperRes(Dataset):

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
        Business card Superresolution Dataset

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
        root_dir = Path(os.environ['BIZCARD_ROOT_DIR'])
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

class BizcardInpaintPartialTrain(BizcardInpaint):

    def __init__(self, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['TrainPartial'] })
        super().__init__(**kwargs)


class BizcardInpaintPartialValidation(BizcardInpaint):

    def __init__(self, num_items:int=1024, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['RandomPartial'] })
        super().__init__(**kwargs)
        self.base = self.base[:num_items]

class BizcardInpaintTrain(BizcardInpaint):

    def __init__(self, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Train'] })
        super().__init__(**kwargs)


class BizcardInpaintValidation(BizcardInpaint):

    def __init__(self, num_items:int=1024, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['Random'] })
        super().__init__(**kwargs)
        self.base = self.base[:num_items]

class BizcardSuperResTrain(BizcardSuperRes):

    def __init__(self, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['TrainSuperRes'] })
        super().__init__(**kwargs)

class BizcardSuperResValidation(BizcardSuperRes):

    def __init__(self, num_items:int=1024, **kwargs):
        if 'names' not in kwargs:
            kwargs.update({ 'names': ['RandomSuperRes'] })
        super().__init__(**kwargs)
        self.base = self.base[:num_items]

def gen_file_list():
    bizcard_root_dir = Path(os.environ['BIZCARD_ROOT_DIR'])
    for sub_dir in bizcard_root_dir.glob('*'):
        if not sub_dir.is_dir():
            continue
        print(f'process {sub_dir.name}')
        flist = []
        for path in tqdm(list(sub_dir.glob('*.jpg'))):
            flist.append(path.name)
        with open(sub_dir / 'flist.pkl', 'wb') as f:
            pickle.dump(flist, f)


def gen_ocr_list():
    bizcard_root_dir = Path(os.environ['BIZCARD_ROOT_DIR'])
    for sub_dir in bizcard_root_dir.glob('*'):
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

def gen_validation():
    validate_src_dir = Path(os.environ['VALIDATE_SRC_DIR'])
    validate_dst_dir = Path(os.environ['VALIDATE_DST_DIR'])
    file_count_all = 0
    for path in list(validate_src_dir.glob('*')):
        file_count_all += 1

    flist = []
    validate_count_max = file_count_all/20
    validate_count = 0
    for path in tqdm(list(validate_src_dir.glob('*.jpg'))):
        flist.append(path.name)
        src_path = os.path.join(validate_src_dir, path.name)
        dst_path = os.path.join(validate_dst_dir, path.name)
        shutil.copy(src_path, dst_path)
        if validate_count > validate_count_max:
            break
        validate_count += 1

    with open(os.path.join(validate_dst_dir, 'flist.pkl'), 'wb') as f:
        pickle.dump(flist, f)

def gen_normalized_img_original_ratio():
    dst_h = dst_w = 512
    normalize_src_dir = Path(os.environ['NORMALIZE_SRC_DIR'])
    normalize_ocr_dir = Path(os.environ['NORMALIZE_OCR_DIR'])
    normalize_dst_dir = Path(os.environ['NORMALIZE_DST_DIR'])
    flist = []
    for path in tqdm(list(normalize_src_dir.glob('*'))):
        # print(path.name)
        ext = os.path.splitext(path.name)[1]
        dst_path = os.path.join(normalize_dst_dir, path.name)
        dst_path = dst_path.replace(ext, ".jpg")

        ocr_path = os.path.join(normalize_ocr_dir, path.name)
        ocr_path = ocr_path.replace(ext, ".json")

        try:
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            angle = ocr_data['analyzeResult']['pages'][0]['angle']

            img_original = cv2.imread(os.path.join(normalize_src_dir, path.name), cv2.IMREAD_LOAD_GDAL)

            if angle > 45 and angle < 135:
                # print('90')
                img = cv2.rotate(img_original, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle < -45 and angle > -135:
                # print('-90')
                img = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)
            elif (angle > 135 and angle < 180) or (angle < -135 and angle > -180):
                # print('180')
                img = cv2.rotate(img_original, cv2.ROTATE_180)
            else:
                # print('0')
                img = img_original
            src_h = img.shape[0]
            src_w = img.shape[1]
            dst_image = np.ones((dst_w, dst_h, 3), dtype=np.uint8)

            ratio = dst_h/max(src_h, src_w)
            dim = ((int)(src_w*ratio), (int)(src_h*ratio))
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            cv2.imwrite(dst_path, resized)
            flist.append(path.name)
        except Exception as e:
            print(f'Scale error: {path.name}')
            print(e)
    with open(os.path.join(normalize_dst_dir, 'flist.pkl'), 'wb') as f:
        pickle.dump(flist, f)

def gen_normalized_img_square():
    dst_h = dst_w = 512
    normalize_src_dir = Path(os.environ['NORMALIZE_SRC_DIR'])
    normalize_ocr_dir = Path(os.environ['NORMALIZE_OCR_DIR'])
    normalize_dst_dir = Path(os.environ['NORMALIZE_DST_DIR'])
    flist = []
    for path in tqdm(list(normalize_src_dir.glob('*'))):
        # print(path.name)
        ext = os.path.splitext(path.name)[1]
        dst_path = os.path.join(normalize_dst_dir, path.name)
        dst_path = dst_path.replace(ext, ".jpg")

        ocr_path = os.path.join(normalize_ocr_dir, path.name)
        ocr_path = ocr_path.replace(ext, ".json")

        try:
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            angle = ocr_data['analyzeResult']['pages'][0]['angle']

            img_original = cv2.imread(os.path.join(normalize_src_dir, path.name), cv2.IMREAD_LOAD_GDAL)

            if angle > 45 and angle < 135:
                # print('90')
                img = cv2.rotate(img_original, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle < -45 and angle > -135:
                # print('-90')
                img = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)
            elif (angle > 135 and angle < 180) or (angle < -135 and angle > -180):
                # print('180')
                img = cv2.rotate(img_original, cv2.ROTATE_180)
            else:
                # print('0')
                img = img_original
            src_h = img.shape[0]
            src_w = img.shape[1]
            dst_image = np.ones((dst_w, dst_h, 3), dtype=np.uint8)

            ratio = dst_h/max(src_h, src_w)
            dim = ((int)(src_w*ratio), (int)(src_h*ratio))
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            dst_image[:resized.shape[0],:resized.shape[1],:] = resized

            cv2.imwrite(dst_path, dst_image)
            flist.append(path.name)
        except Exception as e:
            print(f'Scale error: {path.name}')
            print(e)
    with open(os.path.join(normalize_dst_dir, 'flist.pkl'), 'wb') as f:
        pickle.dump(flist, f)

def gen_partial_img():
    dst_h = dst_w = 128
    pad_ratio = 1.1
    normalize_src_dir = Path(os.environ['NORMALIZE_SRC_DIR'])
    normalize_ocr_dir = Path(os.environ['NORMALIZE_OCR_DIR'])
    normalize_dst_dir = Path(os.environ['NORMALIZE_DST_DIR'])
    flist = []
    for path in tqdm(list(normalize_src_dir.glob('*'))):
        # print(path.name)
        ext = os.path.splitext(path.name)[1]
        dst_path = os.path.join(normalize_dst_dir, path.name)
        dst_path = dst_path.replace(ext, ".jpg")

        ocr_path = os.path.join(normalize_ocr_dir, path.name)
        ocr_path = ocr_path.replace(ext, ".json")

        try:
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)
            angle = ocr_data['analyzeResult']['pages'][0]['angle']
            bbox = ocr_data['analyzeResult']['documents'][0]['fields']['ContactNames']['valueArray'][0]['boundingRegions'][0]['polygon']

            img_original = cv2.imread(os.path.join(normalize_src_dir, path.name), cv2.IMREAD_COLOR)
            if angle > 45 and angle < 135:
                # print('90')
                img = cv2.rotate(img_original, cv2.ROTATE_90_COUNTERCLOCKWISE)
                xmin, xmax = min(bbox[1::2]), max(bbox[1::2])
                ymin, ymax = img_original.shape[1] - max(bbox[0::2]), img_original.shape[1] - min(bbox[0::2])
            elif angle < -45 and angle > -135:
                # print('-90')
                img = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)
                xmin, xmax = img_original.shape[0] - max(bbox[1::2]), img_original.shape[0] - min(bbox[1::2])
                ymin, ymax = min(bbox[0::2]), max(bbox[0::2])
            elif (angle > 135 and angle < 180) or (angle < -135 and angle > -180):
                # print('180')
                img = cv2.rotate(img_original, cv2.ROTATE_180)
                xmin, xmax = img_original.shape[1] - max(bbox[0::2]), img_original.shape[1] - min(bbox[0::2])
                ymin, ymax = img_original.shape[0] - max(bbox[1::2]), img_original.shape[0] - min(bbox[1::2])
            else:
                # print('0')
                img = img_original
                xmin, xmax = min(bbox[0::2]), max(bbox[0::2])
                ymin, ymax = min(bbox[1::2]), max(bbox[1::2])


            x_span, y_span = xmax - xmin, ymax - ymin
            x_center, y_center = (xmax + xmin)//2, (ymax + ymin)//2

            max_src_span = (int)(max(x_span, y_span)*pad_ratio)
            src_x = x_center - max_src_span//2
            src_y = y_center - max_src_span//2
            src_h = img.shape[0]
            src_w = img.shape[1]
            src_x_start = src_x
            src_x_end = src_x + max_src_span
            src_y_start = src_y
            src_y_end = src_y + max_src_span

            if src_x_start < 0:
                src_x_end -= src_x_start
                src_x_start = 0
            elif src_x_end > src_w:
                src_x_start -= (src_x_end - src_w)
                src_x_end = src_w

            if src_y_start < 0:
                src_y_end -= src_y_start
                src_y_start = 0
            elif src_y_end > src_h:
                src_y_start -= (src_y_end - src_h)
                src_y_end = src_h

            if src_x_start < 0 or src_x_end > src_w or src_y_end > src_h or src_y_start < 0:
                print(f"SKIP invalid position: src_x_start={src_x_start}, src_x_end={src_x_end}, src_y_start={src_y_start}, src_y_end={src_y_end}")
                continue

            dst_image = np.ones((dst_w, dst_h, 3), dtype=np.uint8)
            dim = (dst_w, dst_h)
            resized = cv2.resize(img[src_y_start:src_y_end, src_x_start:src_x_end], dim, interpolation = cv2.INTER_AREA)
            dst_image[:resized.shape[0],:resized.shape[1],:] = resized
            cv2.imwrite(dst_path, dst_image)

            flist.append(path.name)
        except Exception as e:
            print(f'Scale error: {path.name}')
            print(e)
            # break

    with open(os.path.join(normalize_dst_dir, 'flist.pkl'), 'wb') as f:
        pickle.dump(flist, f)

def filter_file_list():
    bizcard_root_dir = Path(os.environ['BIZCARD_ROOT_DIR'])
    for sub_dir in bizcard_root_dir.glob('*'):
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
    elif mode == 'gen_ocr':
        gen_ocr_list()
    elif mode == 'gen_normalized_img':
        gen_normalized_img_square()
    elif mode == 'gen_partial_img':
        gen_partial_img()
    elif mode == 'gen_validation':
        gen_validation()


if __name__ == '__main__':
    fire.Fire(main)