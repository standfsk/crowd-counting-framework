from typing import Tuple

from torch import Tensor, nn

from .modules import Backbone, ccsm, Fusion


class FFNet(nn.Module):
    def __init__(self, config: object) -> None:
        super(FFNet, self).__init__()
        self.backbone = config.backbone.lower()
        num_filters = config.num_filters
        self.ccsm1 = ccsm(192, 96, num_filters[0])
        self.ccsm2 = ccsm(384, 192, num_filters[1])
        self.ccsm3 = ccsm(768, 384, num_filters[2])
        self.fusion = Fusion(num_filters[0], num_filters[1], num_filters[2])

        self.backbone_model = Backbone(backbone=self.backbone)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pool1, pool2, pool3 = self.backbone_model(x)

        pool1 = self.ccsm1(pool1)
        pool2 = self.ccsm2(pool2)
        pool3 = self.ccsm3(pool3)
        x = self.fusion(pool1, pool2, pool3)

        B, C, H, W = x.size()
        x_sum = x.contiguous().view(B, -1).sum(1, keepdim=True).view(B, 1, 1, 1)
        x_normed = x.contiguous() / (x_sum + 1e-6)

        return x, x_normed
