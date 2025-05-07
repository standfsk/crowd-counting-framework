# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import logging
from functools import partial
from typing import List, Any

import timm
import torch
from torch import nn, Tensor

logger = logging.getLogger(__name__)

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(
        self, 
        global_pool: bool = False, 
        **kwargs: Any
    ) -> None:
        super(VisionTransformer, self).__init__(**kwargs)
        self.patch_size = kwargs['patch_size']
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(
        self, 
        x: Tensor
    ) -> Tensor:
        B, _, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :]   #.mean(dim=1)
        patch_h, patch_w = H//self.patch_size, W//self.patch_size
        x = x.permute(0, 2, 1).view(B, -1, patch_h, patch_w)

        return x

    def forward(
        self, 
        x: Tensor
    ) -> List[Tensor]:
        x = self.forward_features(x)

        return [x]

    def init_weights(
        self, 
        backbone: str, 
        pretrained: bool
    ) -> None:
        if pretrained:
            pretrained_dict = timm.create_model(backbone, pretrained=pretrained).state_dict()
            logger.info('=> loading pretrained model {}'.format(backbone))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys() and "pos_embed" not in k}
            logger.info("Missing keys: {}".format(list(set(model_dict) - set(pretrained_dict))))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def vit_base_patch16(
    **kwargs: Any
) -> VisionTransformer:
    model = VisionTransformer(
        img_size=(768, 1024), patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(
    **kwargs: Any
) -> VisionTransformer:
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(
    **kwargs: Any
) -> VisionTransformer:
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class MAEBackbone:
    def __init__(
        self, 
        configer: object
    ) -> None:
        self.configer = configer

    def __call__(
        self
    ) -> VisionTransformer:
        arch = self.configer.backbone

        if arch == "vit_base":
            arch_net = vit_base_patch16()
            arch_net.init_weights(backbone="vit_base_patch16_224", pretrained=self.configer.pretrained)
        elif arch == "vit_large":
            arch_net = vit_large_patch16()
            arch_net.init_weights(backbone="vit_large_patch16_224", pretrained=self.configer.pretrained)
        elif arch == "vit_huge":
            arch_net =  vit_huge_patch14()
            arch_net.init_weights(backbone="vit_huge_patch14_224", pretrained=self.configer.pretrained)
        else:
            raise Exception("Architecture undefined!")

        return arch_net