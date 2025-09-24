from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.Tensor.permute(x, self.dims)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def conv(
        in_ch: int,
        out_ch: int,
        ks: int,
        stride: int
) -> nn.Sequential:
    pad = (ks - 1) // 2
    stage = nn.Sequential(nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch, kernel_size=ks, stride=stride,
                                    padding=pad, bias=False),
                          LayerNorm2d((out_ch,), eps=1e-06, elementwise_affine=True),
                          nn.GELU(approximate='none'))
    return stage


class ChannelAttention(nn.Module):
    def __init__(self, channel: int):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        avgout = self.shared_MLP(x)
        return self.sigmoid(avgout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv1(x)
        return self.sigmoid(x)


class ccsm(nn.Module):
    def __init__(self, channel: int, channel2: int, num_filters: int):
        super(ccsm, self).__init__()
        self.ch_att_s = ChannelAttention(channel)
        self.sa_s = SpatialAttention(7)
        self.conv1 = nn.Sequential(
            ODConv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel))
        self.conv2 = nn.Sequential(
            ODConv2d(channel, channel2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel2))

        self.conv3 = nn.Sequential(
            ODConv2d(channel2, channel2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=channel2))
        self.conv4 = nn.Sequential(
            ODConv2d(channel2, num_filters, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_filters))

    def forward(self, x: Tensor) -> Tensor:
        x = self.ch_att_s(x) * x
        pool1 = x
        x = self.conv1(x)
        x = x + pool1
        x = self.conv2(x)
        pool2 = x
        x = self.conv3(x)
        x = x + pool2
        x = self.conv4(x)

        x = self.sa_s(x) * x

        return x


class Fusion(nn.Module):
    def __init__(self, num_filters1: int, num_filters2: int, num_filters3: int):
        super(Fusion, self).__init__()
        self.upsample_1 = nn.ConvTranspose2d(in_channels=num_filters2, out_channels=num_filters2, kernel_size=4,
                                             padding=1, stride=2)
        self.upsample_2 = nn.ConvTranspose2d(in_channels=num_filters3, out_channels=num_filters3, kernel_size=4,
                                             padding=0, stride=4)
        self.final = nn.Sequential(
            nn.Conv2d(num_filters1 + num_filters2 + num_filters3, 1, kernel_size=1, padding=0),
            nn.ReLU(),
        )

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor) -> Tensor:
        x2 = self.upsample_1(x2)
        x3 = self.upsample_2(x3)

        target_size = x1.shape[2:]
        x2 = F.interpolate(x2, size=target_size, mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=target_size, mode="bilinear", align_corners=False)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.final(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int,
            groups: int = 1,
            reduction: float = 0.0625,
            kernel_num: int = 4,
            min_channel: int = 16
    ):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature: float) -> None:
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x: Tensor) -> Tensor:
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x: Tensor) -> Tensor:
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x: Tensor) -> Tensor:
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x: Tensor) -> Tensor:
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            reduction: float = 0.0625,
            kernel_num: int = 4
    ):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups, reduction=reduction,
                                   kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature: float) -> None:
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x: Tensor) -> Tensor:
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.contiguous().reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x: Tensor) -> Tensor:
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
