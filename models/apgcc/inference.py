import os
import time

import cv2
import torch
from core.data import DatasetWithoutLabels
from core.visualization import draw_point_based_result
from tqdm import tqdm

from .model import APGCC


def inference(config: object) -> None:
    t1 = time.time()
    device = "cpu" if config.device == "cpu" else f"cuda:{config.device}"
    os.makedirs(config.save_path, exist_ok=True)

    model = APGCC(config, sync_bn=False, last_pool=False).to(device)
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
            output = model(input_image)
            output_scores = torch.nn.functional.softmax(output["pred_logits"], -1)[:, :, 1][0]
            output_points = output["pred_points"][0]
            pred_points = output_points[output_scores > config.threshold].detach().cpu().numpy().tolist()

            pred_count = (output_scores > config.threshold).detach().cpu().numpy().sum()
            pred_counts.append(pred_count)

            result_image = draw_point_based_result(image=original_image, points=pred_points, count=int(pred_count))
            cv2.imwrite(os.path.join(config.save_path, image_name), result_image)

    t2 = time.time()
    end2end_time = t2 - t1
    fps = round(len(dataset) / end2end_time, 2)

    print(f"Inference done. Total FPS: {fps}")