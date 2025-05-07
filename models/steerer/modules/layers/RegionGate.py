import math
from typing import List, Dict, Tuple, Union, Optional, Any
from torch import nn, Tensor


class SoftGateII(nn.Module):
    """ Soft gating version of FFGate-II"""
    def __init__(self, pool_size: int = 5, channel: int = 10) -> None:
        super(SoftGateII, self).__init__()
        self.pool_size = pool_size
        self.channel = channel

        self.conv1 = conv3x3(channel, channel, stride=2)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU(inplace=True)

        pool_size = math.floor(pool_size / 2 + 0.5)  # for conv stride = 2

        self.avg_layer = nn.AvgPool2d(pool_size)
        self.linear_layer = nn.Conv2d(in_channels=channel, out_channels=2,
                                      kernel_size=1, stride=1)
        self.prob_layer = nn.Softmax()
        self.logprob = nn.LogSoftmax()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.avg_layer(x)
        x = self.linear_layer(x).squeeze()
        softmax = self.prob_layer(x)
        logprob = self.logprob(x)

        x = softmax[:, 1].contiguous()
        x = x.view(x.size(0), 1, 1, 1)
        if not self.training:
            x = (x > 0.5).float()
        return x, logprob