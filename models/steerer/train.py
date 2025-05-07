from typing import Dict, Tuple

import numpy as np
import torch
from core.distributed import barrier
from core.metrics import calculate_metrics
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import reshape_train_data


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
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    loss_info = []
    pred_counts, target_counts = [], []
    for image, target_points, target_density, path, original_image in data_iter:
        input_image = image.to(device)
        target_counts.append([len(p) for p in target_points])
        input_image, target_points = reshape_train_data(input_image, target_points)

        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with autocast(enabled=grad_scaler.is_enabled()):
                    output = model(input_image)
                    loss_dict = loss_fn(output, target_points)
                    pred_density = loss_dict['pred_den']['1']
                    pred_count = pred_density.sum(dim=(1,2,3)).detach().cpu().numpy().tolist()
                    pred_counts.append(pred_count)

                    loss = loss_dict['loss'].mean()
                    loss_info.append(loss.detach().cpu().item())
            else:
                output = model(input_image)
                loss_dict = loss_fn(output, target_points)
                pred_density = loss_dict['pred_den']['1']
                pred_count = pred_density.sum(dim=(1, 2, 3)).detach().cpu().numpy().tolist()
                pred_counts.append(pred_count)

                loss = loss_dict['loss'].mean()
                loss_info.append(loss.detach().cpu().item())

        optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

        barrier(ddp)

    # metric info
    pred_counts = np.array([item for sublist in pred_counts for item in sublist])
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    metric_info = calculate_metrics(pred_counts, target_counts)

    # organize infos
    info = {"loss": round(float(np.mean(loss_info)), 8)}
    info.update(metric_info)

    return model, optimizer, info