# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict, Any, List, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]],
                 eps: float = 1e-6,
                 data_format: str = "channels_last"
        ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# @HEADS.register_module()
class CountingHead(nn.Module):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self, config: Dict[str, Any],
                 init_cfg: Optional[Dict[str, Any]] = None,
                 *args: Any,
                 **kwargs: Any
        ) -> None:
        super(CountingHead, self).__init__()
        self.config = config

        self.out_channels = self.config.out_channels
        self.counter_inchannels = self.config.in_channels

        if self.out_channels <= 0:
            raise ValueError(
                f'num_classes={self.config.out_channels} must be a positive integer')

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.counter_inchannels, self.config.inter_layer[0], 3, stride=1, padding=1, bias=False),
            BatchNorm2d(self.config.inter_layer[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.config.inter_layer[0], self.config.inter_layer[1], kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(self.config.inter_layer[1]),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.config.inter_layer[1], self.config.inter_layer[2], 3, stride=1, padding=1, bias=False),

            BatchNorm2d(self.config.inter_layer[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.config.inter_layer[2], self.config.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        # self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, y: Tensor) -> Tensor:
        return self.decoder(y)

    def pre_logits(self, x: Union[Tensor, Tuple[Tensor, ...]]) -> Tensor:
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x: Union[Tensor, Tuple[Tensor, ...]],
                    softmax: bool = True,
                    post_process: bool = True
        ) -> Union[Tensor, List[float]]:
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x: Tensor,
                      gt_label: Tensor,
                      **kwargs: Any
        ) -> Dict[str, Tensor]:
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

# @HEADS.register_module()
class LocalizationHead(nn.Module):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """
    def __init__(self, config: Dict[str, Any],
                 init_cfg: Optional[Dict[str, Any]] = None,
                 *args: Any,
                 **kwargs: Any
        ) -> None:
        super(LocalizationHead, self).__init__()
        self.config = config

        self.out_channels = self.config.out_channels
        self.counter_inchannels = config.in_channels

        if self.out_channels <= 0:
            raise ValueError(
                f'num_classes={self.config.out_channels} must be a positive integer')

        self.decoder = nn.Sequential(
            # nn.Dropout2d(0.1),
            # ResBlock(in_dim=self.counter_inchannels, out_dim=self.config.inter_layer[0], dilation=0, norm="bn"),
            nn.ConvTranspose2d(self.counter_inchannels, self.config.inter_layer[0], 2, stride=2, padding=0, output_padding=0, bias=False),
            nn.Conv2d( self.config.inter_layer[0], self.config.inter_layer[0], 3, stride=1, padding=1, bias=False),
            BatchNorm2d(self.config.inter_layer[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.config.inter_layer[0], self.config.inter_layer[1], kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(self.config.inter_layer[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.config.inter_layer[1], self.config.inter_layer[2], 3, stride=1, padding=1, bias=False),
            # nn.ConvTranspose2d(self.config.inter_layer[1], self.config.inter_layer[2], 2, stride=2, padding=0, output_padding=0, bias=False),
            BatchNorm2d(self.config.inter_layer[2]),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.config.inter_layer[2], self.config.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, y: Tensor) -> Tensor:
        return self.decoder(y)


    def pre_logits(self, x: Union[Tensor, Tuple[Tensor, ...]]) -> Tensor:
        if isinstance(x, tuple):
            x = x[-1]
        return x

    def simple_test(self, x: Union[Tensor, Tuple[Tensor, ...]],
                    softmax: bool = True,
                    post_process: bool = True
        ) -> Union[Tensor, List[float]]:
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x: Tensor,
                      gt_label: Tensor,
                      **kwargs: Any
        ) -> Dict[str, Tensor]:
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
