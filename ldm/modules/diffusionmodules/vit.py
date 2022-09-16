import torch as th
import torch.nn as nn

from einops import rearrange, repeat
from fairscale.nn.checkpoint import checkpoint_wrapper
from typing import OrderedDict


class QuickGELU(nn.Module):

    def forward(self, x:th.Tensor):
        return x * th.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(
        self,
        hidden_dim:int,
        num_heads:int,
        attn_mask:th.Tensor=None,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(hidden_dim, hidden_dim * 4)),
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(hidden_dim * 4, hidden_dim))
        ]))
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.attn_mask = attn_mask

    def attention(
        self,
        x:th.Tensor,
    ):
        attn_mask = self.attn_mask.to(x) if self.attn_mask else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        x:th.Tensor,
    ):
        h = x + self.attention(self.ln_1(x))
        h = h + self.mlp(self.ln_2(h))
        return h


class ResblockTransformer(nn.Module):
    '''
    transformer with attention block
    '''
    def __init__(
        self,
        hidden_dim:int,
        num_layers:int,
        num_heads:int,
        attn_mask:th.Tensor=None,
        checkpoint:str=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.checkpoint = checkpoint
        blocks = [ResidualAttentionBlock(hidden_dim, num_heads, attn_mask) for _ in range(num_layers)]
        self.blocks = nn.Sequential(*blocks)
        self.apply_actckpt()

    def apply_actckpt(self):
        def use_fairscale_actckpt(module:nn.Module):
            return self.checkpoint == 'fairscale' and isinstance(module, ResidualAttentionBlock)
        for j, module in enumerate(self.blocks):
            if use_fairscale_actckpt(module):
                    self.blocks[j] = checkpoint_wrapper(module)

    def forward(
        self,
        x:th.Tensor,
    ):
        return self.blocks(x)


class TextVisionTransformer(nn.Module):
    '''
    taken from https://github.com/openai/CLIP/blob/main/clip/model.py

    used to encode text line image, with a fixed height but variable width
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
        self.img_height = img_height
        self.out_dim = out_dim
        self.max_patch_len = max_patch_len
        self.checkpoint = checkpoint

        self.conv1 = nn.Conv2d(3, hidden_dim, patch_size, patch_size, bias=False)
        scale = hidden_dim ** -0.5
        self.cls_embed = nn.Parameter(scale * th.randn(hidden_dim))
        max_patches = (img_height // patch_size) * max_patch_len
        self.positional_embedding = nn.Parameter(scale * th.randn(max_patches + 1, hidden_dim))

        self.ln_pre = nn.LayerNorm(hidden_dim)
        self.transformer = ResblockTransformer(hidden_dim, num_layers, num_heads, None, checkpoint)
        self.ln_post = nn.LayerNorm(hidden_dim)
        self.proj = nn.Parameter(scale * th.randn(hidden_dim, out_dim))

    def forward(self, x:th.Tensor):
        h = self.conv1(x)
        h = h[:, :, :, :self.max_patch_len]
        h = rearrange(h, 'b c h w -> b c (h w)')
        h = rearrange(h, 'b c l -> b l c')
        b, l = h.shape[:2]
        cls = repeat(self.cls_embed.to(h), 'c -> b 1 c', b=b)
        h = th.cat([cls, h], dim=1)
        h = h + self.positional_embedding[:l+1]
        h = self.ln_pre(h)
        h = rearrange(h, 'b l c -> l b c')
        h = self.transformer(h)
        h = rearrange(h, 'l b c -> b l c')
        h = self.ln_post(h)
        h = h @ self.proj
        return h
