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
    num_queries: int,
    threshold: float,
) -> Tuple[nn.Module, Optimizer, Dict[str, float]]:
    model.train()
    loss_fn.train()
    data_iter = tqdm(data_loader) if rank == 0 else data_loader
    ddp = nprocs > 1

    loss_info = []
    pred_counts, target_counts = [], []
    for image, target_points, _, path, original_image in data_iter:
        input_image = image.to(device)
        input_image, label = reshape_train_data(input_image, target_points)
        target_counts.append([x['points'].shape[0] for x in label])

        with torch.set_grad_enabled(True):
            if grad_scaler is not None:
                with autocast(enabled=grad_scaler.is_enabled()):
                    output = model(input_image)
                    out_logits = output["pred_logits"]
                    prob = out_logits.sigmoid().view(1, -1, 2)
                    out_logits = out_logits.view(1, -1, 2)
                    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                                           input_image.shape[0] * num_queries, dim=1)
                    for k in range(topk_values.shape[0]):
                        sub_count = topk_values[k, :]
                        sub_count[sub_count < threshold] = 0
                        sub_count[sub_count > 0] = 1
                        sub_count = torch.sum(sub_count).item()
                        pred_counts.append(sub_count)

                    loss_dict = loss_fn(output, label)
                    weight_dict = loss_fn.weight_dict
                    loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    loss_info.append(loss.detach().cpu().item())
            else:
                output = model(input_image)
                out_logits = output["pred_logits"]
                prob = out_logits.sigmoid().view(1, -1, 2)
                out_logits = out_logits.view(1, -1, 2)
                topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                                       input_image.shape[0] * num_queries, dim=1)
                for k in range(topk_values.shape[0]):
                    sub_count = topk_values[k, :]
                    sub_count[sub_count < threshold] = 0
                    sub_count[sub_count > 0] = 1
                    sub_count = torch.sum(sub_count).item()
                    pred_counts.append(sub_count)

                loss_dict = loss_fn(output, label)
                weight_dict = loss_fn.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
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
    pred_counts = np.array(pred_counts)
    target_counts = np.array([sum(item) for item in target_counts])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    metric_info = calculate_metrics(pred_counts, target_counts)

    # organize infos
    info = {"loss_value": round(float(np.mean(loss_info)), 8)}
    info.update(metric_info)

    return model, optimizer, info