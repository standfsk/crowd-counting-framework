from typing import Tuple

from torch import nn, Tensor
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base, convnext_large,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights, ConvNeXt_Large_Weights,
)

CONVNEXT_MODELS = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights),
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights),
    "convnext_large": (convnext_large, ConvNeXt_Large_Weights),
}


class Backbone(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True):
        super(Backbone, self).__init__()

        backbone_model, backbone_weights = CONVNEXT_MODELS[backbone]
        backbone_weights = backbone_weights.DEFAULT if pretrained else None
        feats = list(backbone_model(weights=backbone_weights.IMAGENET1K_V1).features.children())

        self.stem = nn.Sequential(*feats[0:2])
        self.stage1 = nn.Sequential(*feats[2:4])
        self.stage2 = nn.Sequential(*feats[4:6])
        self.stage3 = nn.Sequential(*feats[6:12])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = x.float()
        x = self.stem(x)
        x = self.stage1(x)
        feature1 = x
        x = self.stage2(x)
        feature2 = x
        x = self.stage3(x)

        return feature1, feature2, x
