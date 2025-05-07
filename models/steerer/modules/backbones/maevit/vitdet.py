# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import logging
import math
from functools import partial
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from dict_recursive_update import recursive_update
from timm.layers import drop_path, to_2tuple, trunc_normal_
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint as checkpoint_train

class DropPath(nn.Module):
    def __init__(
        self, 
        drop_prob: Optional[float] = None
    ) -> None:
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(
        self
    ) -> None:
        return 'p={}'.format(self.drop_prob)


class QuickGELU(nn.Module):
    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(
        self, 
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        window_size: Optional[Tuple[int, int]] = None,
        rel_pos_spatial: bool = False
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial = rel_pos_spatial
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.window_size = window_size
        if COMPAT:
            if COMPAT == 2:
                self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size[0] - 1, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size[1] - 1, head_dim))
            else:
                q_size = window_size[0]
                kv_size = q_size
                rel_sp_dim = 2 * q_size - 1
                self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if self.rel_pos_spatial:
            raise
            # attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


def window_partition(
    x: Tensor, 
    window_size: int
) -> Tensor:
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows, 
    window_size: int, 
    H: int, 
    W: int
) -> Tensor:
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
    attn: Tensor,
    q: Tensor,
    q_shape: Tuple[int, int],
    k_shape: Tuple[int, int],
    rel_pos_h: nn.Parameter,
    rel_pos_w: nn.Parameter,
) -> Tensor:
    """
    Spatial Relative Positional Embeddings.

    Source: https://github.com/facebookresearch/mvit/
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(
        self, 
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        rel_pos_spatial: bool = False
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial=rel_pos_spatial

        if COMPAT:
            q_size = window_size[0]
            kv_size = window_size[1]
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, 
        x: Tensor, 
        H: int, 
        W: int
    ) -> Tensor:
        B_, N, C = x.shape
        x = x.reshape(B_, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C

        B_w = x.shape[0]
        N_w = x.shape[1]
        qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)   --> (batchsize, heads, len, head_dim)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        if self.rel_pos_spatial:
            raise

        attn = attn.softmax(dim=-1)
        _attn_mask = (torch.isinf(attn) + torch.isnan(attn))
        attn = attn.masked_fill(_attn_mask, 0)

        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        x = self.proj(x)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)

        return x


class Block(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        window_size: Optional[Tuple[int, int]] = None,
        window: bool = False,
        rel_pos_spatial: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial)
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(
        self, 
        x: Tensor, 
        H: int, 
        W: int
    ) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(
        self, 
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768
    ) -> None:
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # could be dynamic
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self, 
        x: Tensor,
        mask: Optional[Tensor] = None,
        **kwargs: Any
    ) -> Tuple[Tensor, Tuple[int, int], Optional[Tensor]]:
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)

        if mask is not None:
            mask = F.interpolate(mask[None].float(), size=(Hp, Wp)).to(torch.bool)[0]

        return x, (Hp, Wp), mask


class Norm2d(nn.Module):
    def __init__(
        self, 
        embed_dim: int
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage """
    def __init__(
        self, 
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 80,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.,
        norm_layer: Optional[nn.Module] = None,
        window: bool = True,
        use_abs_pos_emb: bool = False,
        interval: int = 3,
        test_pos_mode: str = 'simple_interpolate',
        learnable_pos: bool = False,
        rel_pos_spatial: bool = False,
        lms_checkpoint_train: bool = False,
        pad_attn_mask: bool = False,
        freeze_iters: int = 0,
        act_layer: str = 'GELU',
        pre_ln: bool = False,
        mask_input: bool = False,
        ending_norm: bool = True,
        round_padding: bool = False,
        compat: bool = False
    ) -> None:
        super().__init__()
        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding

        global COMPAT
        COMPAT = compat

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=learnable_pos)
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.patch_shape, cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            raise

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[i], norm_layer=norm_layer,
                window_size=(14, 14) if ((i + 1) % interval != 0) else self.patch_embed.patch_shape,
                window=((i + 1) % interval != 0) if window else False,
                rel_pos_spatial=rel_pos_spatial,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU
            )
            # if self.lms_checkpoint_train == 'fairscale':
            #     block = checkpoint_wrapper(block, offload_to_cpu=True)
            self.blocks.append(block)

        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only
        self.norm = norm_layer(embed_dim)

        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()
        ###
        self.test_pos_mode = test_pos_mode
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.mask_input else None


    def fix_init_weight(
        self
    ) -> None:
        def rescale(
            param: Tensor, 
            layer_id: int
        ) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(
        self, 
        m: nn.Module
    ) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _normalization(
        x: Tensor
    ) -> Tensor:
        assert len(x.shape) == 4
        x = x.sub(torch.tensor([123.675, 116.280, 103.530]).view(1, 3, 1, 1).cuda()).div(torch.tensor([58.395, 57.120, 57.375]).view(1, 3, 1, 1).cuda())
        return x

    def get_num_layers(
        self
    ) -> int:
        return len(self.blocks)

    def forward_features(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None
    ) -> Tensor:
        B, C, H, W = x.shape
        x, (Hp, Wp), mask = self.patch_embed(x, mask)
        batch_size, seq_len, _ = x.size()

        if self.test_pos_mode is False:
            if x.size(1) == self.pos_embed.size(1):
                x = x + self.pos_embed  # BxHWxC
            else: # take top-left if pos_embed > x's dimension
                x = x + self.pos_embed.reshape(1, self.patch_embed.patch_shape[0],
                                               self.patch_embed.patch_shape[1],
                                               self.pos_embed.size(2))[:,:Hp, :Wp, :].reshape(1, x.size(1),
                                                                                              self.pos_embed.size(2))
        elif self.test_pos_mode == 'regenerate':
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (Hp, Wp), cls_token=False)
            x = x + torch.from_numpy(pos_embed).float().unscqueeze(0).cuda()
        elif self.test_pos_mode == 'scaled_regenerate':
            patch_shape = (Hp, Wp)
            orig_size = (math.ceil(Hp/20)*7, math.ceil(Wp/20)*7)

            # as in original scale
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], orig_size, cls_token=False)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).cuda()

            # as in finetuning scale
            pos_embed = pos_embed.reshape(-1, orig_size[0], orig_size[1], self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=(orig_size[0]//7*20, orig_size[1]//7*20),
                                                        mode='bicubic', align_corners=False)

            # as in test image
            pos_embed = pos_embed[:, :, :patch_shape[0], :patch_shape[1]].permute(0, 2, 3, 1).flatten(1, 2)

            x = x + pos_embed
        elif self.test_pos_mode == 'simple_interpolate':
            patch_shape = (Hp, Wp)
            orig_size = (14, 14)

            # as in original scale
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], orig_size, cls_token=False)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).cuda()

            # as in finetuning scale
            pos_embed = pos_embed.reshape(-1, orig_size[0], orig_size[1], self.pos_embed.shape[-1]).permute(0, 3, 1, 2)
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=patch_shape, mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

            x = x + pos_embed
        elif self.test_pos_mode == 'learnable_simple_interpolate':
            patch_shape = (Hp, Wp)
            x = x + get_abs_pos(self.pos_embed, False, patch_shape)
        else:
            raise NotImplementedError

        # x = self.random_masking(x) # effective only if self.mask_input=True (default False), for mask based ssl
        x = self.ln_pre(x)  # effective for clip model only, otherwise nn.Identity

        for i, blk in enumerate(self.blocks):
            # *Warning*: official ckpt implementation leads to NaN loss in many cases, use fairscale if that's the case
            # lms_checkpoint_train = {False, True, 'fairscale'}
            if self.lms_checkpoint_train == True:
                x = checkpoint_train(lambda x: blk(x, Hp, Wp, mask), x, preserve_rng_state=True)
            else:
                x = blk(x, Hp, Wp)

        if self.ending_norm:
            x = self.norm(x)  # b h*w c

        # x = self.unmasking(x)  # effective only if self.mask_input=True (default False), for mask based ssl
        x = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
        x = F.interpolate(x, scale_factor=4)
        return x

    def forward(
        self, 
        input_var: Tensor
    ) -> List[Tensor]:
        output = {}
        x = input_var  #['image']

        # pre_input padding for test support
        # x = self._normalization(x)

        if self.round_padding:
            # pre_input padding for non standard img size support, *** used when test image size varies and not divisible by 32 ***
            stride = self.patch_embed.patch_size
            assert stride[0] == stride[1]
            stride = max(stride[0], self.round_padding)
            output["prepad_input_size"] = [x.shape[-2], x.shape[-1]]  # h, w for sem_seg_postprocess
            target_size = (torch.tensor((x.shape[-1], x.shape[-2])) + (stride - 1)).div(stride, rounding_mode="floor") * stride  # w, h
            padding_size = [  # [l,r,t,b]
                0,
                target_size[0] - x.shape[-1],
                0,
                target_size[1] - x.shape[-2],
                ]
            x = F.pad(x, padding_size, value=0.).contiguous()

        output = self.forward_features(x)
        return [output]

    def init_weights(
        self, 
        pretrained: str = ''
    ) -> None:
        import os
        import logging
        logger = logging.getLogger(__name__)

        if os.path.isfile(pretrained):
            if pretrained.endswith('.tar'):
                pretrained_dict = torch.load(pretrained)['state_dict']

                logger.info('=> loading pretrained model {}'.format(pretrained))
                model_dict = self.state_dict()


                pretrained_dict_filter ={}
                for k, v in pretrained_dict.items():
                    # import pdb
                    # pdb.set_trace()
                    if k[23:] in model_dict.keys() and "pos_embed" not in k:
                        pretrained_dict_filter.update({k[23:]: v})

                #for k, _ in pretrained_dict.items():
                #    logger.info(
                #        '=> loading {} pretrained model {}'.format(k, pretrained))
            elif pretrained.endswith('.pth'):
                pretrained_dict = torch.load(pretrained)['model']#
                logger.info('=> loading pretrained model {}'.format(pretrained))
                model_dict = self.state_dict()


                pretrained_dict_filter ={}
                for k, v in pretrained_dict.items():
                    # import pdb
                    # pdb.set_trace()
                    if k in model_dict.keys() and "pos_embed" not in k:
                        pretrained_dict_filter.update({k: v})
            logger.info(
                "Missing keys: {}".format(list(set(model_dict) - set(pretrained_dict_filter)
                                               )))
            model_dict.update(pretrained_dict_filter)
            # import pdb
            # pdb.set_trace()
            self.load_state_dict(model_dict)

def vit_base_patch16(
    pretrained: bool = False, 
    load_pos_embed: bool = True, 
    **kwargs: Any
) -> ViT:
    default = dict(
        drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    recursive_update(default, kwargs)
    model = ViT(**default)
    return model


def vit_large_patch16(
    pretrained: bool = False, 
    load_pos_embed: bool = True, 
    **kwargs: Any
) -> ViT:
    default = dict(
        drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
        ####
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),

    )
    recursive_update(default, kwargs)
    model = ViT(**default)
    return model

def vit_base_patch16_ema(
    **kwargs: Any
) -> nn.Module:
    backbone = vit_base_patch16(**kwargs)
    backbone.ema = [vit_base_patch16(**kwargs)]
    backbone.ema[0].mask_input = False
    return backbone

class dummy_logger:
    def info(
        self, 
        **kwargs: Any
    ) -> None:
        print(**kwargs)

    def warning(
        self, 
        **kwargs: Any
    ) -> None:
        print(**kwargs)


def clip_checkpoint_preprocess(
    checkpoint: Dict[str, Any]
) -> Dict[str, Any]:
    for k in list(checkpoint.keys()):
        if k.startswith('visual'):
            if k in ["visual.proj", "visual.class_embedding"]:
                new_k = k
            elif k.startswith('visual.transformer.resblocks'):
                new_k = k[len("visual.transformer.res"):]
                new_k = new_k.replace('in_proj_weight', 'qkv.weight')
                new_k = new_k.replace('in_proj_bias', 'qkv.bias')
                new_k = new_k.replace('out_proj', 'proj')
                new_k = new_k.replace('ln_', 'norm')
                new_k = new_k.replace('c_fc', 'fc1')
                new_k = new_k.replace('c_proj', 'fc2')
            else:
                new_k = k[len("visual."):]
                new_k = new_k.replace('positional_embedding', 'pos_embed')
                new_k = new_k.replace('conv1', 'patch_embed.proj')
                new_k = new_k.replace('ln_post', 'norm')
            checkpoint[new_k] = checkpoint[k]
        del checkpoint[k]
    return checkpoint


def load_checkpoint(
    model: nn.Module,
    state_dict: Dict[str, nn.Module],
    load_pos_embed: bool,
    strict: bool = False,
    logger: Optional[logging.Logger] = None
) -> None:
    # get state_dict from checkpoint
    if 'pos_embed' in state_dict:
        if load_pos_embed:
            state_dict['pos_embed'] = interpolate_pos_embed(pos_embed_checkpoint=state_dict['pos_embed'],
                                                            patch_shape=model.patch_embed.patch_shape,
                                                            num_extra_tokens=1)
        else:
            del state_dict['pos_embed']
            print("checkpoint pos_embed removed")

    model_dict = model.state_dict()
    load_dict = {
        k: v for k, v in state_dict.items() if k in model_dict.keys()
    }
    print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))
    load_state_dict(model, state_dict, strict, logger)


def load_state_dict(
    module: nn.Module,
    state_dict: Dict[str, nn.Module],
    strict: bool = False,
    logger: Optional[logging.Logger] = None
) -> None:
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(
        module: nn.Module, 
        prefix: str = ''
    ) -> None:
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')


    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    print("finish load")


def interpolate_pos_embed(
    pos_embed_checkpoint: Tensor,
    patch_shape: Tuple[int, int],
    num_extra_tokens: int
) -> Tensor:
    embedding_size = pos_embed_checkpoint.shape[-1]
    orig_size = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    # class_token and dist_token are kept unchanged
    print(f"Position interpolate from {orig_size} to {patch_shape}")
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] if pos_embed_checkpoint.size(0) == 1 else pos_embed_checkpoint[num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=patch_shape, mode='bicubic', align_corners=False)
    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (b, h*w, c)
    return new_pos_embed


def interpolate_pos_embed_with_cls_token(
    pos_embed_checkpoint: Tensor,
    patch_shape: Tuple[int, int],
    num_extra_tokens: int
) -> Tensor:
    posemb_tok, posemb_grid = (
        pos_embed_checkpoint[:, :num_extra_tokens],
        pos_embed_checkpoint[0, num_extra_tokens:],
    )
    gs_old_h, gs_old_w = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=patch_shape, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, patch_shape[0] * patch_shape[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: List[int],
    cls_token: bool = False
) -> np.ndarray:
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid: np.ndarray
) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: np.ndarray
) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_abs_pos(
    abs_pos: Tensor,
    has_cls_token: bool,
    hw: Tuple[int, int]
) -> Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1).reshape(1, h*w, -1)
    else:
        return abs_pos.reshape(1, h*w, -1)

class MAEvitBackbone(object):
    def __init__(
        self, 
        configer: object
    ) -> None:
        self.configer = configer
        import yaml
        f = open('config.yaml')
        cfg = f.read()

        net_configer = yaml.load(cfg, Loader=yaml.FullLoader)
        self.net_configer = net_configer['common']['backbone']['kwargs']

    def __call__(
        self
    ) -> ViT:
        arch = self.configer.backbone

        if arch == "vit_base":
            arch_net = vit_base_patch16(pretrained=False, load_pos_embed=True)
            # arch_net.init_weights(self.configer.pretrained_backbone)


            if self.configer.pretrained_backbone.endswith('.pt'):
                import io
                # checkpoint = torch.load(self.configer.pretrained_backbone)
                with open(self.configer.pretrained_backbone, 'rb') as f:
                    buffer = io.BytesIO(f.read())
                checkpoint = torch.load(buffer)
                checkpoint = clip_checkpoint_preprocess(checkpoint)

            # load while interpolates position embedding
            load_checkpoint(arch_net, checkpoint, True, strict=False, logger=dummy_logger)
            print('loading clip finish')
            del checkpoint

        elif arch == "vit_large":
            arch_net = vit_large_patch16(pretrained=False, load_pos_embed=True)
            arch_net.init_weights(self.configer.pretrained_backbone)

            # arch_net = ModuleHelper.load_model(
            #     arch_net,
            #     pretrained=self.configer.pretrained_backbone,
            #     all_match=False,
            #     network="hrt_window" if "win" in arch else "hrt",
            # )
        else:
            raise Exception("Architecture undefined!")

        return arch_net