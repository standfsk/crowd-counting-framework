from typing import Any
from typing import List

import torch.nn as nn
from torch import Tensor


# VGG backbone
class Base_VGG(nn.Module):
    def __init__(
        self, 
        name: str, 
        last_pool: bool = False, 
        num_channels: int = 256, 
        **kwargs: Any
    ) -> None:
        super().__init__()
        # loading backbone features
        from .vgg import VGG as vgg
        backbone = vgg(backbone=name, pretrained=True)
        features = list(backbone.encoder.children())

        # setting base module.
        if name == 'vgg16_bn':
            self.body1 = nn.Sequential(*features[:13])
            self.body2 = nn.Sequential(*features[13:23])
            self.body3 = nn.Sequential(*features[23:33])
            if last_pool:
                self.body4 = nn.Sequential(*features[33:44])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[33:43])  # 16x down-sample
        else:
            self.body1 = nn.Sequential(*features[:9])
            self.body2 = nn.Sequential(*features[9:16])
            self.body3 = nn.Sequential(*features[16:23])
            if last_pool:
                self.body4 = nn.Sequential(*features[23:31])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[23:30])  # 16x down-sample
        self.num_channels = num_channels
        self.last_pool = last_pool

    def get_outplanes(
        self
    ) -> List[int]:
        outplanes = []
        for i in range(4):
            last_dims = 0
            for param_tensor in self.__getattr__('body' + str(i + 1)).state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(self.__getattr__('body' + str(i + 1)).state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes  # get the last layer params of all modules, and trans to the size.

    def forward(
        self, 
        tensor_list: Tensor
    ) -> List[Tensor]:
        out = []
        xs = tensor_list
        for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
            xs = layer(xs)
            out.append(xs)
        return out


# ResNet backbone
class Base_ResNet(nn.Module):
    def __init__(
        self, 
        name: str, 
        last_pool: bool = False, 
        num_channels: int = 256, 
        **kwargs: Any
    ) -> None:
        super().__init__()
        print("### ResNet: last_pool=", last_pool)
        # loading backbone features
        from .resnet import resnet18_ibn_a, resnet34_ibn_a, resnet50_ibn_a, resnet101_ibn_a, resnet152_ibn_a
        if name == 'resnet18':
            self.backbone = resnet18_ibn_a(pretrained=True)
        elif name == 'resnet34':
            self.backbone = resnet34_ibn_a(pretrained=True)
        elif name == 'resnet50':
            self.backbone = resnet50_ibn_a(pretrained=True)
        elif name == 'resnet101':
            self.backbone = resnet101_ibn_a(pretrained=True)
        elif name == 'resnet152':
            self.backbone = resnet152_ibn_a(pretrained=True)

        self.num_channels = num_channels
        self.last_pool = last_pool

    def get_outplanes(
        self
    ) -> List[int]:
        outplanes = []
        for Layer in [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]:
            last_dims = 0
            for param_tensor in Layer.state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(Layer.state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes  # get the last layer params of all modules, and trans to the size.

    def forward(
        self, 
        tensor_list: Tensor
    ) -> List[Tensor]:
        xs = tensor_list
        out = self.backbone(xs)
        return out