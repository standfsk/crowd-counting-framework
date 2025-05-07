from typing import Dict, List

import numpy as np
import torch
from core.metrics import calculate_metrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    weight: int
) -> Dict[str, float]:
    model.eval()

    pred_counts, target_counts = [], []
    for image, target_points, _, path, original_image in tqdm(data_loader):
        input_image = image.to(device)
        target_counts.append([len(p) for p in target_points])

        with torch.set_grad_enabled(False):
            output = model(input_image)
            pred_density = output[0] / weight
            pred_count = pred_density.sum().item()
            pred_counts.append(pred_count)

    pred_counts = np.array(pred_counts)
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    return calculate_metrics(pred_counts, target_counts)