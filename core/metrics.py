from typing import Dict

import numpy as np


def calculate_metrics(
    pred_counts: np.ndarray,
    gt_counts: np.ndarray,
) -> Dict[str, float]:
    assert isinstance(pred_counts, np.ndarray), f"Expected numpy.ndarray, got {type(pred_counts)}"
    assert isinstance(gt_counts, np.ndarray), f"Expected numpy.ndarray, got {type(gt_counts)}"
    assert len(pred_counts) == len(gt_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(gt_counts)}"

    # metrics = {
    #     "mae": np.mean(np.abs(pred_counts - gt_counts)),
    #     "rmse": np.sqrt(np.mean((pred_counts - gt_counts) ** 2)),
    # }

    precisions = []
    recalls = []
    accuracys = []
    for pred, gt in zip(pred_counts, gt_counts):
        if pred == 0 and gt == 0:
            precision = 1
            recall = 1
            accuracy = 1
        elif pred == 0:
            precision = 0
            recall = 0
            accuracy = 0
        elif gt == 0 and pred > 0:
            precision = 0
            recall = 0
            accuracy = 0
        else:
            tp = min(gt, pred)
            fp = max(0, pred-gt)
            fn = max(0, gt-pred)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = tp / (tp + fp + fn)
        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)

    mae = np.mean(np.abs(pred_counts - gt_counts))
    rmse = np.sqrt(np.mean((pred_counts - gt_counts) ** 2))
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1_score = 2 * ((precision * recall) / (precision + recall)) if precision and recall else 0.0
    accuracy = np.mean(accuracys)

    metrics = {
        "mae": round(mae, 8),
        "rmse": round(rmse, 8),
        "precision": round(precision, 8),
        "recall": round(recall, 8),
        "f1_score": round(f1_score, 8),
        "accuracy": round(accuracy, 8)
    }
    return metrics