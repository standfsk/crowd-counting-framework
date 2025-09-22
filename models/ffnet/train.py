from typing import Dict, Tuple

import numpy as np
import torch
from core.distributed import barrier, reduce_mean
from core.logging import update_loss_info
from core.metrics import calculate_metrics
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    grad_scaler: GradScaler,
    device: torch.device,
    rank: int,
    nprocs: int,
) -> Tuple[nn.Module, Optimizer, Dict[str, float]]:
    model.train()
    info = None
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    pred_counts, target_counts = [], []
    for image, target_points, target_density, path, original_image in data_iter:
        input_image = image.to(device)
        target_counts.append([len(p) for p in target_points])
        target_points = [p.to(device) for p in target_points]
        target_density = target_density.to(device)
        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with autocast(enabled=grad_scaler.is_enabled()):
                    pred_density, _ = model(input_image)
                    loss, loss_info = loss_fn(pred_density, target_density, target_points)
            else:
                pred_density, _ = model(input_image)
                loss, loss_info = loss_fn(pred_density, target_density, target_points)
            pred_counts.append(pred_density.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist())

        optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # loss info
        loss_info = {k: reduce_mean(v.detach(), nprocs).item() if ddp else v.detach().item() for k, v in loss_info.items()}
        info = update_loss_info(info, loss_info)
        barrier(ddp)

    # metric info
    pred_counts = np.array([item for sublist in pred_counts for item in sublist])
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    metric_info = calculate_metrics(pred_counts, target_counts)

    # organize infos
    info = {k: round(float(np.mean(v)), 8) for k,v in info.items()}
    info.update(metric_info)

    return model, optimizer, info