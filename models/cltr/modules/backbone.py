# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchvision
from core.utils import NestedTensor
from torch import nn, Tensor
from torchvision.models import get_model_weights
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(
        self, 
        n: int
    ) -> None:
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, 
        state_dict: Dict[str, Tensor],
        prefix: str,
        local_metadata: Dict[str, int],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str]
    ) -> None:
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        train_backbone: bool, 
        num_channels: int, 
    ) -> None:
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(
        self, 
        tensor_list: NestedTensor
    ) -> Dict[str, NestedTensor]:
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(
        self, 
        name: str,
        train_backbone: bool,
        dilation: bool
    ) -> None:
        weights_enum = get_model_weights(name)
        weights = weights_enum.DEFAULT if weights_enum is not None else None

        backbone = getattr(torchvision.models, name)(
            weights=weights,
            replace_stride_with_dilation=[False, False, dilation],
            norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(
        self, 
        backbone: Backbone, 
        position_embedding: nn.Module
    ) -> None:
        super().__init__(backbone, position_embedding)

    def forward(
        self, 
        tensor_list: NestedTensor
    ) -> Tuple[List[NestedTensor], List]:
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(
    backbone: str,
    hidden_dim: int,
    position_embedding: str,
    lr_backbone: float,
    dilation: bool
) -> nn.Module:
    position_embedding = build_position_encoding(hidden_dim=hidden_dim, position_embedding=position_embedding)
    train_backbone = lr_backbone > 0
    backbone = Backbone(backbone, train_backbone, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
