from typing import Dict

import numpy as np
import torch
from core.metrics import calculate_metrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import reshape_eval_data


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_queries: int,
    threshold: float,
    crop_size: int
) -> Dict[str, float]:
    model.eval()

    pred_counts, target_counts = [], []
    for image, target_points, _, path, original_image in tqdm(data_loader):
        input_image = image.to(device)
        target_counts.append([len(p) for p in target_points])
        input_image, label = reshape_eval_data(input_image, target_points, crop_size)

        with torch.set_grad_enabled(False):
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

    pred_counts = np.array(pred_counts)
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    return calculate_metrics(pred_counts, target_counts)