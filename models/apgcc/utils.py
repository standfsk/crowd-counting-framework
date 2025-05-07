from typing import Tuple

import torch
import torch.nn as nn
from losses import apgcc_loss


def get_loss_fn(config: object) -> nn.Module:
    return apgcc_loss(config)

def get_optimizer(config: object, model: nn.Module) -> Tuple:
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.Adam(param_dicts, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    return optimizer, scheduler