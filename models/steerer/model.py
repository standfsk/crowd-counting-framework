from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .modules import BackboneSelector, HeadSelector, upsample_module
from .utils import freeze_model


class STEERER(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.resolution_num = config.resolution_num
        self.counter_type = config.counter_type

        self.backbone_model = BackboneSelector(config).get_backbone()

        if self.counter_type == 'withMOE':
            self.multi_counter = HeadSelector(config).get_head()
            self.head_counter = HeadSelector(config).get_head()
            freeze_model(self.head_counter)
            self.upsample_module = upsample_module(config)

        elif self.counter_type == 'single_resolution':
            self.head_counter = HeadSelector(config).get_head()
        else:
            raise ValueError('counter_type must be basleline or withMOE')

    def forward(self, x: Tensor) -> List[Tensor]:
        if self.counter_type == "single_resolution":
            features = self.backbone_model(x)
            feat0_h, feat0_w = features[0].size(2), features[0].size(3)
            upsampled_features = [features[0]]
            for i in range(1, len(features)):
                upsampled_features.append(F.upsample(features[i], size=(feat0_h, feat0_w), mode='bilinear'))
            upsampled_features = torch.cat(upsampled_features, 1)
            output = self.head_counter(upsampled_features)

        elif self.counter_type == "withMOE":
            self.head_counter.load_state_dict(self.multi_counter.state_dict())
            freeze_model(self.head_counter)
            features = self.backbone_model(x)
            features = features[self.resolution_num[0]:self.resolution_num[-1] + 1]
            output = self.upsample_module(features, self.multi_counter, self.head_counter)

        return output

