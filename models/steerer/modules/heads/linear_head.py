# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict, List, Tuple, Any, Union

import torch.nn.functional as F
from torch import nn, Tensor

from .cls_head import ClsHead
from ..builder import HEADS


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self, num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[Dict[str, Any]] = None,
                 *args: Any,
                 **kwargs: Any
        ) -> None:
        init_cfg = init_cfg or dict(type="Normal", layer="Linear", std=0.01)
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

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

    def forward_train(self, x: Union[Tensor, Tuple[Tensor, ...]],
                      gt_label: Tensor,
                      **kwargs: Any
        ) -> Dict[str, Tensor]:
        x = self.pre_logits(x)
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses
