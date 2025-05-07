from typing import List

import timm
import torch
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
        self.encoder = self.assemble_modules(list(encoder.children()))
        self.start_idx = 2
        self.end_idx = len(self.encoder)

    def assemble_modules(self, model: List[nn.Module]) -> nn.ModuleList:
        model_ = nn.ModuleList()
        counter = 0
        while counter < len(model):
            mod = model[counter]
            if isinstance(mod, nn.MaxPool2d):
                model_.append(mod)
                counter += 1
            else:
                assert isinstance(mod, nn.Conv2d)
                block = nn.ModuleList([mod])
                for i in range(counter + 1, len(model)):
                    mod = model[i]
                    if isinstance(mod, nn.BatchNorm2d):
                        block.append(mod)
                    if isinstance(mod, nn.ReLU):
                        block.append(mod)
                        break
                model_.append(nn.Sequential(*block))
                counter = i + 1
        return model_

    def forward(self, x: Tensor) -> List[Tensor]:
        feats = []
        for idx, mod in enumerate(self.encoder):
            x = mod(x)
            if self.start_idx <= idx < self.end_idx:
                feats.append(torch.clone(x))
        return feats
