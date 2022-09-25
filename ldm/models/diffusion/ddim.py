"""SAMPLING ONLY."""

import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torch import Tensor
from typing import Any, Callable, Dict, List, Union

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
)


class DDIMSampler(object):

    def __init__(
        self,
        model:nn.Module,
        schedule:str="linear",
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(
        self,
        name:str,
        attr:Tensor,
    ):
        model_device = next(self.model.parameters()).device
        if type(attr) == Tensor and attr.device != model_device:
            attr = attr.to(model_device)
        setattr(self, name, attr)

    def make_schedule(
        self,
        ddim_num_steps:int,
        ddim_discretize:str="uniform",
        ddim_eta:float=0.,
        verbose:bool=True
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1.-ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(
        self,
        S:int,
        batch_size:int,
        shape:List[int],
        conditioning:Union[Tensor,Dict]=None,
        callback:Callable[[int],None]=None,
        img_callback:Callable[[Tensor,int],None]=None,
        quantize_x0:bool=False,
        eta:float=0.,
        mask:Tensor=None,
        x0:Tensor=None,
        temperature:float=1.,
        noise_dropout:float=0.,
        score_corrector:Any=None,
        corrector_kwargs:Dict=None,
        verbose:bool=True,
        x_T:Tensor=None,
        log_every_t:int=100,
        unconditional_guidance_scale:float=1.,
        unconditional_conditioning:Tensor=None,
        dynamic_thresholding:float=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dynamic_thresholding=dynamic_thresholding)

        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond:Tensor,
        shape:List[int],
        x_T:Tensor=None,
        ddim_use_original_steps:bool=False,
        callback:Callable[[int],None]=None,
        timesteps:np.ndarray=None,
        quantize_denoised:bool=False,
        mask:Tensor=None,
        x0:Tensor=None,
        img_callback:Callable[[Tensor,int],None]=None,
        log_every_t:int=100,
        temperature:float=1.,
        noise_dropout:float=0.,
        score_corrector:Any=None,
        corrector_kwargs:Dict=None,
        unconditional_guidance_scale:float=1.,
        unconditional_conditioning:Tensor=None,
        dynamic_thresholding:float=None,
    ):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {
            'x_inter': [img],
            'pred_x0': [img],
        }
        time_range = reversed(range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        rank = torch.distributed.get_rank()
        disable = rank not in (-1, 0)
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps, disable=disable)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised, temperature=temperature,
                noise_dropout=noise_dropout, score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,)

            img, pred_x0 = outs

            if dynamic_thresholding is not None:
                img_cpu = img.cpu().numpy()
                s = np.percentile(
                    np.abs(img_cpu), dynamic_thresholding,
                    axis=tuple(range(1, img_cpu.ndim)))
                s = np.max(s[...,None], axis=-1, initial=1.0)
                for i in range(b):
                    img_cpu[i] = np.clip(img_cpu[i], -s[i], s[i]) / s[i]
                img = torch.from_numpy(img_cpu).to(img.device)

            if callback:
                callback(i)

            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        if mask is not None:
            img = x0 * mask + (1. - mask) * img

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x:Tensor,
        c:Tensor,
        t:Tensor,
        index:int,
        repeat_noise:bool=False,
        use_original_steps:bool=False,
        quantize_denoised:bool=False,
        temperature:float=1.,
        noise_dropout:float=0.,
        score_corrector:Any=None,
        corrector_kwargs:Dict=None,
        unconditional_guidance_scale:float=1.,
        unconditional_conditioning:Union[Tensor,Dict]=None,
    ):
        b, *_, device = *x.shape, x.device

        use_cfg = not (unconditional_conditioning is None or unconditional_guidance_scale == 1.)
        if not use_cfg:
            model_out = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if isinstance(c, Dict):
                assert unconditional_conditioning.keys() == c.keys()
                uc = unconditional_conditioning
                keys = list(c.keys())
                c_in = {}
                for key in keys:
                    if isinstance(c[key], str):
                        c_in[key] = c[key]
                    elif isinstance(c[key], Tensor):
                        c_in[key] = torch.cat([uc[key], c[key]])
                    elif isinstance(c[key], list):
                        c_in[key] = []
                        for i, val in enumerate(c[key]):
                            if isinstance(val, Tensor):
                                c_in[key].append(torch.cat([uc[key][i], c[key][i]]))
                            elif isinstance(val, str):
                                c_in[key].append(val)
                            else:
                                raise ValueError()
                    else:
                        raise ValueError()
            else:
                c_in = torch.cat([unconditional_conditioning, c])
            model_out_uncond, model_out = self.model.apply_model(x_in, t_in, c_in).chunk(2)

        if self.model.var_parameterization == 'learned_range':
            ch = x.shape[1]
            model_mean_out, _ = torch.split(model_out, ch, dim=1)
            if use_cfg:
                model_mean_out_uncond, _ = torch.split(model_out_uncond, ch, dim=1)
        else:
            model_mean_out = model_out
            if use_cfg:
                model_mean_out_uncond = model_out_uncond

        if self.model.mean_parameterization == 'eps':
            if use_cfg:
                e_t = model_mean_out_uncond + unconditional_guidance_scale * (model_mean_out - model_mean_out_uncond)
            else:
                e_t = model_mean_out
        elif self.model.mean_parameterization == 'x0':
            if use_cfg:
                e_t = self.model.noise_from_predict_start(x, t, model_mean_out)
                e_t_uncond = self.model.noise_from_predict_start(x, t, model_mean_out_uncond)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            else:
                e_t = self.model.noise_from_predict_start(x, t, model_mean_out)

        if score_corrector is not None:
            assert self.model.mean_parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
