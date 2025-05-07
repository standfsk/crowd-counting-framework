import os
from time import time

import cv2
import numpy as np
import torch
from core.data import get_dataloader
from core.evaluation import sliding_window_predict
from core.metrics import calculate_metrics
from core.visualization import draw_density_based_result
from tqdm import tqdm

from .model import _clip_ebc


def test(config: object) -> None:
    t1 = time.time()
    device = "cpu" if config.device == "cpu" else f"cuda:{config.device}"

    if config.save:
        os.makedirs(os.path.join(config.save_path, "vis"), exist_ok=True)

    if config.log:
        test_log = open(os.path.join(config.save_path, "log.txt"), "w")
        test_log.write(f"name,gt,pred\n")

    config.bins = [[0.0, 0.0], [1.0, 1.0], [2.0, float("inf")]]
    config.anchor_points = [0.0, 1.0, 2.10737]

    if config.sliding_window:
        assert config.window_size is not None, f"Window size must be provided when sliding_window is True, but got {args.window_size}"
        assert config.stride is not None, f"Stride must be provided when sliding_window is True, but got {args.stride}"

    model = _clip_ebc(config).to(device)
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    data_loader = get_dataloader(config, split="test", ddp=False)
    pred_counts, target_counts = [], []
    for image, target_points, _, path, original_image in tqdm(data_loader):
        image_name = os.path.basename(path[0])
        original_image = original_image[0]
        input_image = image.to(device)
        target_count = [len(p) for p in target_points][0]
        target_counts.append(target_count)

        with torch.set_grad_enabled(False):
            if config.sliding_window:
                pred_density = sliding_window_predict(model, input_image, config.window_size, config.stride)
            else:
                pred_density = model(input_image)

            pred_count = pred_density.detach().cpu().numpy().sum()
            pred_counts.append(pred_count)
            if config.save:
                density_image, result_image = draw_density_based_result(image=original_image, density_map=pred_density, count=int(pred_count))
                cv2.imwrite(os.path.join(config.save_path, "vis", image_name), result_image)

            if config.log:
                test_log.write(f"{image_name},{target_count},{pred_count}\n")

    t2 = time.time()
    test_log.close()
    end2end_time = t2 - t1
    fps = round(len(data_loader) / end2end_time, 2)

    pred_counts = np.array(pred_counts)
    target_counts = np.array(target_counts)
    assert len(pred_counts) == len(target_counts), f"Length of predictions and ground truths should be equal, but got {len(pred_counts)} and {len(target_counts)}"

    metrics = calculate_metrics(pred_counts, target_counts)
    metrics['fps'] = fps

    with open(os.path.join(config.save_path, "result.txt"), "w") as result_txt:
        for key, value in metrics.items():
            result_txt.write(f"{key}: {value}\n")