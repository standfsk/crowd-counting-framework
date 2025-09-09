from typing import List, Tuple, Dict, Union

import os
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import vgg, FeatureFuser, ChannelReducer, ConvNormActivation


class FusionCount(nn.Module):
    def __init__(self, config: object) -> None:
        super(FusionCount, self).__init__()
        self.backbone = config.backbone.lower()
        self.regression = config.regression
        self.bins = config.bins
        self.anchor_points = config.anchor_points
        if os.path.exists(config.state_dict):
            self.state_dict = config.state_dict
        else:
            self.state_dict = None

        self.backbone_model = vgg(backbone=self.backbone, pretrained=True, state_dict=self.state_dict)
        self.backbone_config = self.get_channel_list(self.backbone_model.model_cfgs[self.backbone])

        batch_norm = False
        if "_bn" in self.backbone:
            batch_norm = True

        self.fuser_1 = FeatureFuser(self.backbone_config[0], batch_norm=batch_norm)
        self.fuser_2 = FeatureFuser(self.backbone_config[1], batch_norm=batch_norm)
        self.fuser_3 = FeatureFuser(self.backbone_config[2], batch_norm=batch_norm)
        self.fuser_4 = FeatureFuser(self.backbone_config[3], batch_norm=batch_norm)

        self.reducer_1 = ChannelReducer(in_channels=64, out_channels=32, dilation=2, batch_norm=batch_norm)
        self.reducer_2 = ChannelReducer(in_channels=128, out_channels=64, dilation=2, batch_norm=batch_norm)
        self.reducer_3 = ChannelReducer(in_channels=256, out_channels=128, dilation=2, batch_norm=batch_norm)
        self.reducer_4 = ChannelReducer(in_channels=512, out_channels=256, dilation=2, batch_norm=batch_norm)

        if self.regression:
            self.bins = None
            self.anchor_points = None
            self.output_layer = ConvNormActivation(in_channels=32, out_channels=1, kernel_size=1, stride=1, dilation=1,
                                                   norm_layer=None, activation_layer=nn.ReLU(inplace=True))
        else:
            assert len(self.bins) == len(self.anchor_points), f"Expected bins and anchor_points to have the same length, got {len(self.bins)} and {len(self.anchor_points)}"
            assert all(len(b) == 2 for b in self.bins), f"Expected bins to be a list of tuples of length 2, got {self.bins}"
            assert all(bin[0] <= p <= bin[1] for bin, p in zip(self.bins, self.anchor_points)), f"Expected anchor_points to be within the range of the corresponding bin, got {self.bins} and {self.anchor_points}"

            self.anchor_points = torch.tensor(self.anchor_points, dtype=torch.float32, requires_grad=False).view(1, -1, 1, 1)
            self.output_layer = ConvNormActivation(in_channels=32, out_channels=len(self.bins), kernel_size=1, stride=1,
                                                   dilation=1, norm_layer=None)

    def get_channel_list(self, model_cfg: Dict[str, List[Union[int, str]]]) -> List[List[int]]:
        result = []
        current_group = []
        for item in model_cfg:
            if item == 'M':
                if current_group:
                    result.append(current_group)
                current_group = [current_group[-1]] if current_group else []
            else:
                current_group.append(int(item))  # Convert to int if needed
        if current_group:
            result.append(current_group)
        return result[1:]

    def group_features(self, features: nn.ModuleList) -> List[nn.ModuleList]:
        features_group = []
        start = 0
        for channel_group in self.backbone_config:
            features_group.append(features[start:start + len(channel_group)])
            start += len(channel_group)
        return features_group

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        features = self.backbone_model(x)
        feat_1, feat_2, feat_3, feat_4 = self.group_features(features)

        feat_1 = self.fuser_1(feat_1)
        feat_2 = self.fuser_2(feat_2)
        feat_3 = self.fuser_3(feat_3)
        feat_4 = self.fuser_4(feat_4)

        feat_4 = self.reducer_4(feat_4)
        feat_4 = F.interpolate(feat_4, size=feat_3.shape[-2:], mode="bilinear", align_corners=False)

        feat_3 = feat_3 + feat_4
        feat_3 = self.reducer_3(feat_3)
        feat_3 = F.interpolate(feat_3, size=feat_2.shape[-2:], mode="bilinear", align_corners=False)

        feat_2 = feat_2 + feat_3
        feat_2 = self.reducer_2(feat_2)
        feat_2 = F.interpolate(feat_2, size=feat_1.shape[-2:], mode="bilinear", align_corners=False)

        feat_1 = feat_1 + feat_2
        feat_1 = self.reducer_1(feat_1)
        feat_1 = F.interpolate(feat_1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        output = self.output_layer(feat_1)
        mu = F.avg_pool2d(output, kernel_size=8, stride=8)

        if self.regression:
            B, C, H, W = mu.size()
            mu_sum = mu.view([B, -1]).sum().unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            mu_normed = mu / (mu_sum + 1e-6)
            return mu, mu_normed
        else:
            probs = mu.softmax(dim=1)
            exp = (probs * self.anchor_points.to(mu.device)).sum(dim=1, keepdim=True)
            if self.training:
                return mu, exp
            else:
                return exp
