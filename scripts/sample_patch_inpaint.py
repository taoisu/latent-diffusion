import argparse
import os
import torch

import numpy as np
import torch as th
import torch.nn as nn

from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from pathlib import Path

from PIL import Image

rescale = lambda x: (x + 1.) / 2.


def make_batch(
    image_path: Path,
    mask_path: Path,
    patch_path: Path,
    ldm: nn.Module,
    device: th.device,
):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = np.array(img).astype(np.uint8)
    img = (img/127.5-1.0).astype(np.float32)
    size = ldm.image_size
    h, w, c = img.shape[:3]
    assert h == w == size
    assert c == 3

    mask = Image.open(mask_path)
    mask = mask.convert('L')
    mask = np.array(mask).astype(np.uint8)
    mask = (mask/255).astype(np.float32)[:, :, None]
    h, w, c = mask.shape[:3]
    assert h == w == size
    assert c == 1

    patch = Image.open(patch_path)
    embder = ldm.cond_stage_model.preprocess
    patch = embder(patch)
    patch = rearrange(patch, 'c h w -> h w c')

    mask_img = img*(1-mask)
    mask = mask*2-1.0
    batch = {
        'image': torch.from_numpy(img[None]).to(device),
        'masked_image': torch.from_numpy(mask_img[None]).to(device),
        'patch': patch[None].to(device),
        'mask': torch.from_numpy(mask[None]).to(device),
    }
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=Path,
        required=True,
        help="path to checkpoint",
    )
    parser.add_argument(
        "-i",
        "--indir",
        type=Path,
        required=True,
        help="directory containing examples",
    )
    parser.add_argument(
        '-o',
        '--outdir',
        type=Path,
        required=True,
        help="directory to write results to",
    )
    parser.add_argument(
        '-s',
        '--steps',
        type=int,
        default=50,
        help='number of ddim sampling steps'
    )
    opt = parser.parse_args()

    mask_paths = opt.indir.glob('*_mask.png')
    dataset = []
    for mask_path in mask_paths:
        image_name = mask_path.name.replace('_mask.png', '.png')
        image_path = opt.indir / image_name
        assert image_path.exists()
        patch_name = f'{image_name[:-len(".png")]}_patch'
        patch_paths = list(opt.indir.glob(f'{patch_name}_*.png'))
        dataset.append({
            'mask_path': mask_path,
            'image_path': image_path,
            'patch_paths': patch_paths,
        })

    config = OmegaConf.load('configs/just-diffusion/pin128.yaml')
    model = instantiate_from_config(config.model)
    model.load_state_dict(
        torch.load(opt.resume, map_location=torch.device('cpu'))['state_dict'],
        strict=False)
    device = torch.device('cpu')
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for example in dataset:
                mask_path = example['mask_path']
                image_path = example['image_path']
                patch_paths = example['patch_paths']
                for patch_path in patch_paths:
                    batch = make_batch(image_path, mask_path, patch_path, model, device)
                    c = model.cond_stage_model.encode(batch)
                    sz, ch = model.image_size, model.channels
                    shape = (ch, sz, sz)
                    samples_ddim, intermediates = sampler.sample(
                        S=opt.steps,
                        conditioning=c,
                        batch_size=1,
                        shape=shape,
                        verbose=False,
                    )
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    image = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
                    mask = torch.clamp((batch["mask"]+1.0)/2.0, min=0.0, max=1.0)
                    predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                    predicted_image = rearrange(predicted_image, 'b c h w -> b h w c')
                    inpainted_image = (1-mask)*image+mask*predicted_image

                    patch_name_noext = patch_path.name[:-len('.png')]
                    out_pred_path = opt.outdir / f'{patch_name_noext}.pred.png'
                    predicted_image = predicted_image.to(th.device('cpu')).numpy()[0]*255
                    predicted_image = predicted_image.astype(np.uint8)
                    Image.fromarray(predicted_image).save(out_pred_path)

                    out_inpaint_path = opt.outdir / f'{patch_name_noext}.inpaint.png'
                    inpainted_image = inpainted_image.to(th.device('cpu')).numpy()[0]*255
                    inpainted_image = inpainted_image.astype(np.uint8)
                    Image.fromarray(inpainted_image).save(out_inpaint_path)

                    for k, vs in intermediates.items():
                        for i, v in enumerate(vs):
                            out_path = opt.outdir / f'{patch_name_noext}.{k}.{i}.png'
                            v = rearrange(v, 'b c h w -> b h w c')
                            v = torch.clamp((v+1.0)/2.0, min=0.0, max=1.0)
                            v = v.to(th.device('cpu')).numpy()[0]*255
                            v = v.astype(np.uint8)
                            Image.fromarray(v).save(out_path)


