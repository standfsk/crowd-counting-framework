import os
import time

import cv2
import numpy as np
import torch
from core.data import get_dataloader
from core.metrics import calculate_metrics
from core.visualization import draw_point_based_result
from tqdm import tqdm

from .model import CLTR
from .utils import reshape_eval_data, get_points


def test(config: object) -> None:
    t1 = time.time()
    device = "cpu" if config.device == "cpu" else f"cuda:{config.device}"

    if config.save:
        os.makedirs(os.path.join(config.save_path, "vis"), exist_ok=True)

    if config.log:
        test_log = open(os.path.join(config.save_path, "log.txt"), "w")
        test_log.write(f"name,gt,pred\n")

    model = CLTR(config).to(device)
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
        input_image, label = reshape_eval_data(input_image, target_points, config.crop_size)

        with torch.set_grad_enabled(False):
            output = model(input_image)
            out_logits, out_point = output["pred_logits"], output["pred_points"]

            pred_points = get_points(original_image, out_logits, out_point, config.crop_size, config.num_queries)
            pred_count = len(pred_points)
            pred_counts.append(pred_count)
            if config.save:
                result_image = draw_point_based_result(image=original_image, points=pred_points, count=int(pred_count))
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