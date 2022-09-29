import fire
import os
import torch

import PIL

from copy import deepcopy
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import (
    LightningModule as LitModel,
    LightningDataModule as LitData,
)
from torch import Tensor
from torch.utils.data import default_collate
from typing import Dict, List, Union

from PIL import Image

from ldm.util import instantiate_from_config, log_txt_as_img


def load_model_from_config(config:Dict, resume:str):
    pl_sd = torch.load(resume, map_location='cpu')
    model = instantiate_from_config(config)
    model.load_state_dict(pl_sd['state_dict'], strict=False)
    model.cuda()
    model.eval()
    return model


def pil_aware_collate_fn(batch):
    ret = {}
    for item in batch:
        keys = []
        for k in item.keys():
            if isinstance(item[k], PIL.Image.Image):
                if k not in ret:
                    ret[k] = []
                ret[k].append(item[k])
                keys.append(k)
        for k in keys:
            del item[k]
    ret.update(default_collate(batch))
    return ret


def tsr2img(tsr:Tensor, p:int):
    b, c, h, w = tsr.shape
    imgs = torch.clamp((tsr+1)*127.5, 0, 255).to(torch.uint8)
    imgs = rearrange(imgs, 'b c h w -> b h w c')
    imgs = imgs.cpu().numpy()
    canvas = Image.new('RGB', (w, b*(h+p)), 'black')
    for i, img in enumerate(imgs):
        img = img if c == 3 else img[...,0]
        canvas.paste(Image.fromarray(img), (0, i*(h+p)))
    return canvas


def save_images(
    key:str,
    image:Tensor,
    mask:Tensor,
    text:List[str],
    samples:Tensor,
    inter:Union[List, Dict],
    out_dir:str,
):
    b, _, h, w = samples.shape
    assert isinstance(inter, (list, dict))
    if isinstance(inter, dict):
        inter = inter['x_inter']
    num_inter = len(inter)
    p = 1
    hs, ws = b*(h+p), (num_inter+4)*(w+p)
    canvas = Image.new('RGB', (ws, hs), 'black')
    canvas.paste(tsr2img(image, p), (0*(w+p), 0))
    canvas.paste(tsr2img(mask*2-1, p), (1*(w+p), 0))
    canvas.paste(tsr2img(samples, p), (2*(w+p), 0))
    canvas.paste(tsr2img(log_txt_as_img((h, w), text), p), (3*(w+p), 0))
    for i in range(num_inter):
        x, y = (i+4)*(w+p), 0
        inter_tsr = inter[i]
        canvas.paste(tsr2img(inter_tsr, p), (x, y))
    canvas.save(f'{out_dir}/{key}.jpg')


@torch.no_grad()
def sample(
    model:LitModel,
    data:LitData,
    out_dir:str,
):
    data_loader = data.val_dataloader()
    data_loader.collate_fn = pil_aware_collate_fn
    num = 16
    for batch in data_loader:
        batch = { k: v.cuda() if isinstance(v, Tensor) else v for k, v in batch.items() }
        image, text = batch['image'], batch['text']
        assert batch['image'].shape[0] >= num
        z, c, x, xrec, xc = model.get_input(
            batch,
            model.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=num)
        mask = 1 - rearrange(batch['mask'][:num], 'b h w c -> b c h w')

        batch_uncond = deepcopy(batch)
        batch_uncond['text'] = ['' for _ in batch_uncond['text']]
        batch_uncond['txt_image'] = [ PIL.Image.new(img.mode, img.size, 'white') for img in batch['txt_image'] ]
        _, uncond_c = model.get_input(
            batch_uncond,
            model.first_stage_key,
            force_c_encode=True,
            bs=num)

        image = rearrange(image, 'b h w c -> b c h w')

        # ddpm
        with model.ema_scope('Plot text inpaint - ddpm'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=False,
                ddim_steps=None,
                x0=z[:num],
                mask=mask)
            save_images('ddpm', image[:num], mask[:num], text[:num], samples, inters, out_dir)

        # ddpm cg (classifier guidance) 2.0
        with model.ema_scope('Plot text inpaint - ddpm - cg 2.0'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=False,
                ddim_steps=None,
                x0=z[:num],
                mask=mask,
                unconditional_guidance_scale=2.0,
                unconditional_conditioning=uncond_c)
            save_images('ddpm_cg2', image, mask, text, samples, inters, out_dir)

        # ddpm cg 5.0
        with model.ema_scope('Plot text inpaint - ddpm - cg 5.0'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=False,
                ddim_steps=None,
                x0=z[:num],
                mask=mask,
                unconditional_guidance_scale=5.0,
                unconditional_conditioning=uncond_c)
            save_images('ddpm_cg5', image, mask, text, samples, inters, out_dir)

        # ddpm cg 10.0
        with model.ema_scope('Plot text inpaint - ddpm - cg 10.0'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=False,
                ddim_steps=None,
                x0=z[:num],
                mask=mask,
                unconditional_guidance_scale=10.0,
                unconditional_conditioning=uncond_c)
            save_images('ddpm_cg10', image, mask, text, samples, inters, out_dir)

        # ddim 250
        with model.ema_scope('Plot text inpaint - ddim 250'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=True,
                ddim_steps=250,
                x0=z[:num],
                mask=mask)
            save_images('ddim_step250', image, mask, text, samples, inters, out_dir)

        # ddim 500
        with model.ema_scope('Plot text inpaint - ddim 500'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=True,
                ddim_steps=500,
                x0=z[:num],
                mask=mask)
            save_images('ddim_step500', image, mask, text, samples, inters, out_dir)

        # ddim 250 - cg 2.0
        with model.ema_scope('Plot text inpaint - ddim 250 - cg 2.0'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=True,
                ddim_steps=250,
                x0=z[:num],
                mask=mask,
                unconditional_guidance_scale=2.0,
                unconditional_conditioning=uncond_c)
            save_images('ddim_step250_cg2', image, mask, text, samples, inters, out_dir)

        # ddim 250 - cg 5.0
        with model.ema_scope('Plot text inpaint - ddim 250 - cg 5.0'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=True,
                ddim_steps=250,
                x0=z[:num],
                mask=mask,
                unconditional_guidance_scale=5.0,
                unconditional_conditioning=uncond_c)
            save_images('ddim_step250_cg5', image, mask, text, samples, inters, out_dir)

        # ddim 250 - cg 10.0
        with model.ema_scope('Plot text inpaint - ddim 250 - cg 10.0'):
            samples, inters = model.sample_log(
                cond=c,
                batch_size=num,
                ddim=True,
                ddim_steps=250,
                x0=z[:num],
                mask=mask,
                unconditional_guidance_scale=10.0,
                unconditional_conditioning=uncond_c)
            save_images('ddim_step250_cg10', image, mask, text, samples, inters, out_dir)
            break


def main(
    base:str,
    resume:str,
    out_dir:str,
):
    init_method="env://?rank=0&world_size=1"
    torch.distributed.init_process_group('NCCL', init_method)

    print('load config')
    config = OmegaConf.load(base)

    print('load model')
    model = load_model_from_config(config.model, resume)
    model.clip_denoised = True

    print('load data')
    config.data.params.train.params.names = ['Random']
    config.data.params.num_workers = 1
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    print('sample data')
    os.makedirs(out_dir, exist_ok=True)
    sample(model, data, out_dir)


if __name__ == '__main__':
    fire.Fire(main)