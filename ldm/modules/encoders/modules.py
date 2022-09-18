import clip
import kornia
import torch

import numpy as np
import torch as th
import torch.nn as nn

from einops import rearrange
from functools import partial
from torch import Tensor
from typing import Dict, List, OrderedDict
from einops import rearrange, repeat

from transformers import AutoTokenizer
from transformers import ViTModel, BeitModel, CLIPVisionModel, Swinv2Model
from transformers.models.t5.modeling_t5 import T5Block, T5EncoderModel

from PIL import Image

from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from ldm.modules.diffusionmodules.vit import TextVisionTransformer


class AbstractEncoder(nn.Module):

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):

    def __init__(self, embed_dim:int, n_classes:int=1000, key:str='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch:Tensor, key:str=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):

    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class IdentityEmbedder(AbstractEncoder):

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, batch:Dict):
        h = batch['patch'][:,None,:]
        return h

    @torch.no_grad()
    def encode(self, batch:Dict):
        h = self(batch)
        concat_h = th.cat([batch['masked_image'], batch['mask']], dim=-1)
        concat_h = rearrange(concat_h, 'b h w c -> b c h w')
        return {
            'c_concat': concat_h,
            'c_crossattn': h,
        }

class ClipPatchEmbedder(AbstractEncoder):

    def __init__(
        self,
        clip_model_name:str,
    ):
        super().__init__()
        model, preprocess = clip.load(clip_model_name)
        self.model = model
        self.model.visual.ret_patch = True
        self.preprocess = preprocess

    def forward(self, batch:Dict):
        patch = batch['patch']
        h_pool, h_patch = self.model.encode_image(patch)
        mask = torch.count_nonzero(patch, dim=list(range(1, patch.ndim))) == 0
        h_pool.masked_fill_(mask[(...,)+(None,)*(h_pool.ndim-1)], 0.)
        h_patch.masked_fill_(mask[(...,)+(None,)*(h_patch.ndim-1)], 0.)
        return h_pool, h_patch

    @torch.no_grad()
    def encode(self, batch:Dict):
        h_pool, h_patch = self(batch)
        concat_h = th.cat([batch['masked_image'], batch['mask']], dim=-1)
        concat_h = rearrange(concat_h, 'b h w c -> b c h w')
        return {
            'c_concat': concat_h,
            'c_crossattn': h_patch,
            'c_emb': h_pool,
            'c_name': 'patch',
        }


class BERTTokenizer(AbstractEncoder):

    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(
        self,
        device:str="cuda",
        vq_interface:bool=True,
        max_length:int=77,
    ):
        super().__init__()
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text:str):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text:str):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text:str):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed:int,
        n_layer:int,
        vocab_size:int=30522,
        max_seq_len:int=77,
        device:str="cuda",
        use_tokenizer:bool=True,
        embedding_dropout:float=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(
                vq_interface=False,
                max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout)

    def forward(self, text:str):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text:str):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):

    def __init__(
        self,
        n_stages=1,
        method='bilinear',
        multiplier=0.5,
        in_channels=3,
        out_channels=None,
        bias=False,
    ):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class TextImageEmbedder(AbstractEncoder):
    '''
    Embed text line image w/ ViT style encoder
    '''
    def __init__(
        self,
        img_height:int,
        patch_size:int,
        hidden_dim:int,
        num_layers:int,
        num_heads:int,
        out_dim:int,
        max_patch_len:int=128,
        checkpoint:str=None,
    ):
        super().__init__()
        self.model = TextVisionTransformer(
            img_height, patch_size, hidden_dim, num_layers, num_heads, out_dim, max_patch_len, checkpoint
        )

    def forward(self, x:torch.Tensor):
        return self.model(x)

    def encode(self, cond:Dict):
        txt_images = cond['txt_image']
        max_width = max(txt_img.size[0] for txt_img in txt_images)
        txt_imgs_tsr = []
        for txt_image in txt_images:
            _, height = txt_image.size
            txt_image_pad = Image.new(txt_image.mode, (max_width, height), 'white')
            txt_image_pad.paste(txt_image, (0, 0))
            txt_img_tsr = np.array(txt_image_pad).astype(np.uint8)
            txt_imgs_tsr.append(txt_img_tsr)
        txt_imgs_tsr = np.stack(txt_imgs_tsr)
        txt_imgs_tsr = (txt_imgs_tsr/127.5-1.0).astype(np.float32)
        txt_imgs_tsr = torch.from_numpy(txt_imgs_tsr).to(cond['image'])
        txt_imgs_tsr = rearrange(txt_imgs_tsr, 'b h w c -> b c h w')
        outputs = self(txt_imgs_tsr)
        cls_hidden_state = outputs[:, 0, :]
        return {
            'c_crossattn': outputs,
            'c_emb': cls_hidden_state,
            'c_name': 'txtimg',
        }


class TextImageInpaintEmbedder(TextImageEmbedder):
    '''
    Embed text with vit transformer
    '''
    def encode(self, cond:Dict):
        ret = super().encode(cond)
        mask = cond['mask']
        ret.update({ 'c_mask': rearrange(mask, 'b h w c -> b c h w') })
        return ret


class FrozenPretrainedImageEmbedder(AbstractEncoder):
    '''
    Use pretrained huggingface transformer vision encoder

    Support the following models:
    * CLIPVisionModel
    * ViTModel (google/vit-*)
    * Swinv2Model
    * BeitModel
    '''
    def __init__(
        self,
        model_name:str,
        image_size:int=None,
        canvas_pad:int=12,
        with_concat:bool=True,
        with_emb:bool=True,
        with_crossattn:bool=True,
        with_mask:bool=True,
    ):
        super().__init__()
        if model_name.startswith('google/vit-'):
            model_class = ViTModel
        elif model_name.startswith('openai/clip-'):
            model_class = CLIPVisionModel
        elif model_name.startswith('microsoft/swinv2-'):
            model_class = Swinv2Model
        elif model_name.startswith('microsoft/beit-'):
            model_class = BeitModel
        else:
            raise ValueError()
        self.model = model_class.from_pretrained(model_name)
        self.canvas_pad = canvas_pad
        self.with_concat = with_concat
        self.with_emb = with_emb
        self.with_crossattn = with_crossattn
        self.with_mask = with_mask
        self.image_size = image_size

    def prep_text_cond(self, cond:Dict):
        txt_images = cond['txt_image']
        imgs_tsr, cc_imgs_tsr = [], []
        cc_img_size, canvas_pad = self.image_size, self.canvas_pad
        image_size = self.model.config.image_size
        hf_img_size = image_size // 2
        canvas_size = image_size - canvas_pad * 2
        for txt_image in txt_images:
            w, h = txt_image.size
            l = max(w, h)
            scale = canvas_size / l
            if scale < 1.:
                w, h = int(w * scale), int(h * scale)
                txt_image = txt_image.resize((w, h), Image.Resampling.BICUBIC)
            cond_image = Image.new('RGB', (image_size, image_size), 'white')
            cond_image.paste(txt_image, (hf_img_size - w // 2, hf_img_size - h // 2))
            cond_image_tsr = np.array(cond_image).astype(np.uint8)
            imgs_tsr.append(cond_image_tsr)
            cc_cond_image = cond_image.resize((cc_img_size, cc_img_size), Image.Resampling.BICUBIC)
            cc_cond_image = cc_cond_image.convert('L')
            cc_cond_image_tsr = np.array(cc_cond_image)[..., np.newaxis].astype(np.uint8)
            cc_imgs_tsr.append(cc_cond_image_tsr)
        imgs_tsr = np.stack(imgs_tsr)
        imgs_tsr = (imgs_tsr/127.5-1.0).astype(np.float32)
        imgs_tsr = torch.from_numpy(imgs_tsr).to(cond['image'])
        imgs_tsr = rearrange(imgs_tsr, 'b h w c -> b c h w')
        cc_imgs_tsr = np.stack(cc_imgs_tsr)
        cc_imgs_tsr = (cc_imgs_tsr/127.5-1.0).astype(np.float32)
        cc_imgs_tsr = torch.from_numpy(cc_imgs_tsr).to(cond['image'])
        cc_imgs_tsr = rearrange(cc_imgs_tsr, 'b h w c -> b c h w')
        return imgs_tsr, cc_imgs_tsr

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def encode(self, cond:Dict):
        imgs_tsr, cc_imgs_tsr = self.prep_text_cond(cond)
        outputs = self.model(imgs_tsr)
        ret = OrderedDict()
        if self.with_concat:
            ret.update({ 'c_concat': cc_imgs_tsr })
        if self.with_emb:
            ret.update({ 'c_emb': outputs.pooler_output })
        if self.with_crossattn:
            ret.update({ 'c_crossattn': outputs.last_hidden_state })
        if self.with_mask:
            ret.update({ 'c_mask': rearrange(cond['mask'], 'b h w c -> b c h w') })
        ret.update({ 'c_name': 'txtimg' })
        return ret


class FrozenPretrainedTextEmbedder(AbstractEncoder):
    '''
    Use pretrained huggingface transformer language encoder
    '''
    def __init__(
        self,
        model_name:str,
        max_len:int=128,
    ):
        super().__init__()
        if model_name.startswith('google/t5'):
            self.model = T5EncoderModel.from_pretrained(model_name, low_cpu_mem_usage=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.fsdp_cls = (T5Block,)
            self.ckpt_cls = (T5Block,)
        else:
            raise ValueError()
        self.max_len = max_len

    def fsdp_wrap_policy(
        self,
        module: nn.Module,
        recurse: bool,
        **kwargs,
    ) -> bool:
        return True if recurse else isinstance(module, self.fsdp_cls)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def encode(self, text:List[str]):
        batch_size = len(text)
        max_len = self.max_len
        inputs = self.tokenizer(
            text,
            padding='longest',
            return_tensors='pt',
            max_length=max_len,
            truncation=True)
        device = self.root_device if hasattr(self, 'root_device') else self.model.device
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        outputs = self(**inputs)
        last_hidden_state = outputs.last_hidden_state
        eos_hidden_state_pos = inputs['attention_mask'].sum(dim=1) - 1
        eos_hidden_state = last_hidden_state[
            torch.arange(0, batch_size, device=device),
            eos_hidden_state_pos,
        ]
        return {
            'c_crossattn': last_hidden_state,
            'c_crossattn_mask': inputs['attention_mask'],
            'c_emb': eos_hidden_state,
            'c_name': 'caption',
        }

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return {}


class FrozenTextInpaintEmbedder(FrozenPretrainedTextEmbedder):
    '''
    Embed text with pretrained transformer, and 
    '''
    def __init__(
        self,
        concat:bool=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.concat = concat

    def encode(self, cond:Dict):
        ret = super().encode(cond['text'])
        mask, image = cond['mask'], cond['image']
        if self.concat:
            cond_image = image.clone()
            cond_image[mask.broadcast_to(cond_image.shape) == 1] = -1
            cond_image = rearrange(cond_image, 'b h w c -> b c h w')
            ret.update({ 'c_concat': cond_image })
        ret.update({ 'c_mask': rearrange(mask, 'b h w c -> b c h w') })
        return ret


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
    Uses the CLIP image encoder.
    """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))

