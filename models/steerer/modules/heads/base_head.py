from abc import ABCMeta, abstractmethod
from typing import Optional, Dict, Any, List, Union

from torch import nn, Tensor


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base head."""

    def __init__(self, init_cfg: Optional[Dict[str, Any]] = None) -> None:
        super(BaseHead, self).__init__()
        self.init_cfg = init_cfg

    @abstractmethod
    def forward_train(self, x: Tensor, gt_label: Union[Tensor, List[Tensor]], **kwargs: Any) -> None:
        pass
