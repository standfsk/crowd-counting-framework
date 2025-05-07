from typing import List

import timm
from torch import nn, Tensor

model_cfgs = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "vgg11_bn": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "vgg13_bn": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
    "vgg16_bn": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
    "vgg19_bn": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]
}

class VGG(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True) -> None:
        super(VGG, self).__init__()
        self.model_cfgs = model_cfgs
        encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(-2,))
        self.encoder = encoder

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.encoder(x)[-1]
        return x
