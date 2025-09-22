import math
from functools import partial
from typing import Tuple

import torch
from losses import dmloss, daceloss
from torch import nn


def get_loss_fn(config: object) -> nn.Module:
    if config.regression:
        assert config.weight_ot is not None and config.weight_tv is not None, f"Expected weight_ot and weight_tv to be not None, got {config.weight_ot} and {config.weight_tv}"
        loss_fn = dmloss(
            input_size=config.input_size,
            reduction=config.reduction,
        )
    else:
        loss_fn = daceloss(
            bins=config.bins,
            reduction=config.reduction,
            weight_count_loss=config.weight_count_loss,
            count_loss_type=config.count_loss_type,
            input_size=config.input_size,
        )
    return loss_fn


def cosine_annealing_warm_restarts(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
    T_0: int,
    T_mult: int,
    eta_min: float,
) -> float:
    """
    Learning rate scheduler.
    The learning rate will linearly increase from warmup_lr to lr in the first warmup_epochs epochs.
    Then, the learning rate will follow the cosine annealing with warm restarts strategy.
    """
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    assert isinstance(warmup_epochs,
                      int) and warmup_epochs >= 0, f"warmup_epochs must be non-negative, got {warmup_epochs}."
    assert isinstance(warmup_lr, float) and warmup_lr > 0, f"warmup_lr must be positive, got {warmup_lr}."
    assert isinstance(T_0, int) and T_0 >= 1, f"T_0 must be greater than or equal to 1, got {T_0}."
    assert isinstance(T_mult, int) and T_mult >= 1, f"T_mult must be greater than or equal to 1, got {T_mult}."
    assert isinstance(eta_min, float) and eta_min > 0, f"eta_min must be positive, got {eta_min}."
    assert isinstance(base_lr, float) and base_lr > 0, f"base_lr must be positive, got {base_lr}."
    assert base_lr > eta_min, f"base_lr must be greater than eta_min, got base_lr={base_lr} and eta_min={eta_min}."
    assert warmup_lr >= eta_min, f"warmup_lr must be greater than or equal to eta_min, got warmup_lr={warmup_lr} and eta_min={eta_min}."

    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
    else:
        epoch -= warmup_epochs
        if T_mult == 1:
            T_cur = epoch % T_0
            T_i = T_0
        else:
            n = int(math.log((epoch / T_0 * (T_mult - 1) + 1), T_mult))
            T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** (n)

        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2

    return lr / base_lr

def get_optimizer(config: object, model: nn.Module) -> Tuple:
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=partial(
            cosine_annealing_warm_restarts,
            warmup_epochs=config.warmup_epochs,
            warmup_lr=config.warmup_lr,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            base_lr=config.lr
        ),
    )

    return optimizer, scheduler