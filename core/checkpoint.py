import os
import sys
from collections import OrderedDict
from typing import Union, Tuple, Dict, List

import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

def load_checkpoint(
    config: object,
    model: nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    grad_scaler: GradScaler,
) -> Tuple[nn.Module, Adam, Union[LambdaLR, None], GradScaler, int, Union[Dict[str, float], None], Dict[str, List[float]], Dict[str, float]]:
    ckpt_path = os.path.join(config.save_path, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        loss_info = ckpt["loss_info"]
        hist_scores = ckpt["hist_scores"]
        best_scores = ckpt["best_scores"]

        if scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if grad_scaler is not None:
            grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])

        print(f"Loaded checkpoint from {ckpt_path}.")

    else:
        start_epoch = 1
        loss_info, hist_scores = None, {k: torch.inf for k in ["mae", "rmse", "precision", "recall", "f1_score", "accuracy"]}
        best_scores = {k: torch.inf for k in hist_scores.keys()}
        print(f"Checkpoint not found at {ckpt_path}.")

    return model, optimizer, scheduler, grad_scaler, start_epoch, loss_info, hist_scores, best_scores


def save_checkpoint(
    epoch: int,
    model_state_dict: OrderedDict[str, Tensor],
    optimizer_state_dict: OrderedDict[str, Tensor],
    scheduler_state_dict: OrderedDict[str, Tensor],
    loss_info: Dict[str, List[float]],
    hist_scores: Dict[str, List[float]],
    best_scores: Dict[str, float],
    ckpt_dir: str,
    grad_scaler_state_dict: OrderedDict[str, Tensor],
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
        "loss_info": loss_info,
        "hist_scores": hist_scores,
        "best_scores": best_scores,
    }
    torch.save(ckpt, os.path.join(ckpt_dir, "ckpt.pt"))