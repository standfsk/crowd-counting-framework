from typing import Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .modules import vgg, _init_weights


class DMCount(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.backbone = config.backbone.lower()
        self.regression = config.regression
        self.bins = config.bins
        self.anchor_points = config.anchor_points

        self.backbone_model = vgg(backbone=self.backbone, pretrained=True)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.reg_layer.apply(_init_weights)

        if self.regression:
            self.bins = None
            self.anchor_points = None
            self.output_layer = nn.Sequential(
                nn.Conv2d(128, 1, 1),
                nn.ReLU(inplace=True)
            )
        else:
            assert len(self.bins) == len(self.anchor_points), f"Expected bins and anchor_points to have the same length, got {len(self.bins)} and {len(self.anchor_points)}"
            assert all(len(b) == 2 for b in self.bins), f"Expected bins to be a list of tuples of length 2, got {self.bins}"
            assert all(bin[0] <= p <= bin[1] for bin, p in zip(self.bins, self.anchor_points)), f"Expected anchor_points to be within the range of the corresponding bin, got {self.bins} and {self.anchor_points}"

            self.anchor_points = torch.tensor(self.anchor_points, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)
            self.output_layer = nn.Conv2d(128, len(self.bins), kernel_size=1)

        self.output_layer.apply(_init_weights)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x =  self.backbone_model(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = self.reg_layer(x)
        x = self.output_layer(x)

        if self.regression:
            return x
        else:
            probs = x.softmax(dim=1)
            exp = (probs * self.anchor_points.to(x.device)).sum(dim=1, keepdim=True)
            if self.training:
                return x, exp
            else:
                return exp

