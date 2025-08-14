import os
import time

import cv2
import torch
from core.data import DatasetWithoutLabels
from core.visualization import draw_density_based_result
from tqdm import tqdm

from .model import DMCount


def inference(config: object) -> None:
    t1 = time.time()
    device = "cpu" if config.device == "cpu" else f"cuda:{config.device}"
    os.makedirs(config.save_path, exist_ok=True)

    config.bins = [[0.0, 0.0], [1.0, 1.0], [2.0, float("inf")]]
    config.anchor_points = [0.0, 1.0, 2.10737]

    model = DMCount(config).to(device)
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = DatasetWithoutLabels(data_path=config.data_path, input_size=config.input_size)
    pred_counts = []
    for image, original_image, data_path in tqdm(dataset):
        image_name = os.path.basename(data_path)
        original_image = original_image[0]
        input_image = image.to(device)

        with torch.set_grad_enabled(False):
            pred_density = model(input_image)
            pred_count = pred_density.detach().cpu().numpy().sum()
            pred_counts.append(pred_count)

            density_image, result_image = draw_density_based_result(image=original_image, density_map=pred_density, count=int(pred_count))
            cv2.imwrite(os.path.join(config.save_path, image_name), result_image)

    t2 = time.time()
    end2end_time = t2 - t1
    fps = round(len(dataset) / end2end_time, 2)

    print(f"Inference done. Total FPS: {fps}")