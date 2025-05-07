from typing import Dict

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
    threshold: float = 0.5
) -> Dict[str, float]:
    model.eval()

    pred_counts, target_counts = [], []
    for image, target_points, _, path, original_image in tqdm(data_loader):
        input_image = image.to(device)
        target_counts.append([len(p) for p in target_points])

        with torch.set_grad_enabled(False):
            output = model(input_image)
            output_scores = torch.nn.functional.softmax(output["pred_logits"], -1)[:, :, 1]
            pred_counts.append((output_scores > threshold).sum(dim=1).detach().cpu().numpy().tolist())

    pred_counts = np.array([item for sublist in pred_counts for item in sublist])
    target_counts = np.array([item for sublist in target_counts for item in sublist])
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"
    return calculate_metrics(pred_counts, target_counts)