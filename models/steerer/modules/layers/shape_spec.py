# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Dict, Tuple, Union, Optional, Any
from torch import nn, Tensor
from collections import namedtuple


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to obtain the shape inference ability among pytorch modules.

    Attributes:
        channels:
        height:
        width:
        stride:
    """
    def __new__(cls, *,
                channels: Optional[int] = None,
                height: Optional[int] = None,
                width: Optional[int] = None,
                stride: Optional[int] = None
        ) -> None:
        return super().__new__(cls, channels, height, width, stride)
