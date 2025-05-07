from typing import List, Tuple, Optional

import numpy as np
import torch
from losses import steerer_loss
from timm.scheduler import CosineLRScheduler, StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from torch import nn, Tensor


def get_loss_fn(config: object) -> nn.Module:
    return steerer_loss(config)

def get_optimizer(config: object, model: nn.Module, n_iter_per_epoch: int) -> Tuple:
    # optimizer
    param_dicts = [{"params": model.parameters(), "lr": config.lr}]
    if config.optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(param_dicts,
                                    momentum=config.momentum,
                                    nesterov=config.nesterov,
                                    lr=config.lr,
                                    weight_decay=config.weight_decay)
    elif config.optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_dicts,
                                      eps=config.eps,
                                      betas=config.betas,
                                      lr=config.lr,
                                      weight_decay=config.weight_decay)
    else:
        raise ValueError("Wrong optimizer selected")

    # scheduler
    num_steps = int(config.epochs * n_iter_per_epoch)
    warmup_steps = int(config.warmup_epochs * n_iter_per_epoch)
    decay_steps = int(config.decay_epochs * n_iter_per_epoch)
    if config.scheduler_type.lower() == "cosine":
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=num_steps,
                                      lr_min=config.min_lr,
                                      warmup_lr_init=config.warmup_lr,
                                      warmup_t=warmup_steps,
                                      cycle_limit=1,
                                      t_in_epochs=False)
    elif config.scheduler_type.lower() == "linear":
        scheduler = LinearLRScheduler(optimizer,
                                      t_initial=num_steps,
                                      lr_min_rate=0.01,
                                      warmup_lr_init=config.warmup_lr,
                                      warmup_t=warmup_steps,
                                      t_in_epochs=False)
    elif config.scheduler_type.lower() == "step":
        scheduler = StepLRScheduler(optimizer,
                                    decay_t=decay_steps,
                                    decay_rate=config.decay_rate,
                                    warmup_lr_init=config.warmup_lr,
                                    warmup_t=warmup_steps,
                                    t_in_epochs=False)
    else:
        raise ValueError("Wrong scheduler selected")

    return optimizer, scheduler

class LinearLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t: int) -> List[float]:
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [
                v - ((v - v * self.lr_min_rate) * (t / total_t))
                for v in self.base_values
            ]
        return lrs

    def get_epoch_values(self, epoch: int) -> Optional[List[float]]:
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int) -> Optional[List[float]]:
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

def freeze_model(model: nn.Module) -> None:
    for (name, param) in model.named_parameters():
            param.requires_grad = False

def update_label(label: Tensor, point: Tensor, shape: List[int], scale: int) -> Tensor:
    w_idx = np.clip(int((point[0] / scale).round()), 0, shape[1] // scale - 1)
    h_idx = np.clip(int((point[1] / scale).round()), 0, shape[0] // scale - 1)
    label[h_idx, w_idx] += 1
    return label

def reshape_train_data(image: Tensor, label: Tensor) -> Tuple[Tensor, List[Tensor]]:
    height = image.shape[2]
    width = image.shape[3]
    shape = [height, width]

    labels = []
    for points in label:
        labelx1 = torch.zeros((height, width), dtype=torch.float32)
        labelx2= torch.zeros((height // 2 , width // 2), dtype=torch.float32)
        labelx4 = torch.zeros((height // 4, width // 4), dtype=torch.float32)
        labelx8 = torch.zeros((height // 8, width // 8), dtype=torch.float32)
        for point in points:
            labelx1 = update_label(labelx1, point, shape, scale=1)
            labelx2 = update_label(labelx2, point, shape, scale=2)
            labelx4 = update_label(labelx4, point, shape, scale=4)
            labelx8 = update_label(labelx8, point, shape, scale=8)
        labels.append([labelx1, labelx2, labelx4, labelx8])

    if image.shape[0] > 1:
        labels = [torch.stack(values, dim=0).to(image.device) for values in zip(*labels)]
    else:
        labels = [x.unsqueeze(0).to(image.device) for x in labels[0]]
    return image, labels