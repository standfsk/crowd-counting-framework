import math
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from core.distributed import barrier, reduce_dict
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
    clip_max_norm: float = 0.1,
    threshold: float = 0.5
) -> Tuple[nn.Module, Optimizer, Dict[str, float]]:
    model.train()
    loss_fn.train()
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    loss_info = {"loss_value": [], "loss_ce": [], "loss_points": [], "loss_ce_scaled": [], "loss_points_scaled": []}
    pred_counts, target_counts = [], []
    for image, target_points, _, path, original_image in data_iter:
        input_image = image.to(device)
        target_counts.append([len(p) for p in target_points])
        target_points = [{'point': p.to(device), 'labels': torch.ones(p.shape[0], dtype=int, device=device)} for p in target_points]
        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with autocast(enabled=grad_scaler.is_enabled()):
                    output = model(input_image)
                    output_scores = torch.nn.functional.softmax(output["pred_logits"], -1)[:, :, 1]
                    pred_counts.append((output_scores > threshold).sum(dim=1).detach().cpu().numpy().tolist())

                    loss_dict = loss_fn(output, target_points)
                    weight_dict = loss_fn.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                    loss_dict_reduced = reduce_dict(loss_dict, nprocs)
                    for k, v in loss_dict_reduced.items():
                        loss_info[k].append(v.item())
                        loss_info[k+"_scaled"].append(v.item() * weight_dict[k])
                    loss_value = sum(loss_info["loss_ce_scaled"] + loss_info["loss_points_scaled"])
                    loss_info["loss_value"].append(loss_value)

                    if not math.isfinite(loss_value):
                        print(f"Loss is {loss_value}, stopping training")
                        sys.exit(1)
            else:
                output = model(input_image)
                output_scores = torch.nn.functional.softmax(output["pred_logits"], -1)[:, :, 1]
                pred_counts.append((output_scores > threshold).sum(dim=1).detach().cpu().numpy().tolist())

                loss_dict = loss_fn(output, target_points)
                weight_dict = loss_fn.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                loss_dict_reduced = reduce_dict(loss_dict, nprocs)
                for k, v in loss_dict_reduced.items():
                    loss_info[k].append(v.item())
                    loss_info[k + "_scaled"].append(v.item() * weight_dict[k])
                loss_value = sum(loss_info["loss_ce_scaled"] + loss_info["loss_points_scaled"])
                loss_info["loss_value"].append(loss_value)

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    sys.exit(1)

        optimizer.zero_grad()
        if clip_max_norm > 0:
           torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        if grad_scaler is not None:
            grad_scaler.scale(losses).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            losses.backward()
            optimizer.step()

        barrier(ddp)

    # metric info
    pred_counts = np.array([item for sublist in pred_counts for item in sublist])
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    metric_info = calculate_metrics(pred_counts, target_counts)

    # organize infos
    info = {k: round(float(np.mean(v)), 8) for k,v in loss_info.items()}
    info.update(metric_info)

    return model, optimizer, info