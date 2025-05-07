import math
from typing import Tuple, List, Dict

import cv2
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from losses import cltr_loss
from torch import nn, Tensor


def get_loss_fn(config: object) -> nn.Module:
    return cltr_loss(config)

def get_optimizer(config: object, model: nn.Module) -> Tuple:
    param_dicts = [
        {"params": model.parameters(), "lr": config.lr},
    ]

    optimizer = torch.optim.Adam(param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.lr_step], gamma=0.1, last_epoch=-1)
    return optimizer, scheduler

def calculate_knn_distance(gt_points: np.ndarray, num_point: int, num_knn: int = 4) -> Tensor:

    if num_point >= 4:
        tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
        distances, locations = tree.query(gt_points, k=min(num_knn, num_point))
        distances = np.delete(distances, 0, axis=1)
        distances = np.mean(distances, axis=1)
        distances = torch.from_numpy(distances).unsqueeze(1)

    elif num_point == 0:
        distances = gt_points.clone()[:, 0].unsqueeze(1)

    elif num_point == 1:
        tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
        distances, locations = tree.query(gt_points, k=num_point)
        distances = torch.from_numpy(distances).unsqueeze(1)

    elif num_point == 2:
        tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
        distances, locations = tree.query(gt_points, k=num_point)
        distances = np.delete(distances, 0, axis=1)
        distances = (distances[:, 0]) / 1.0
        distances = torch.from_numpy(distances).unsqueeze(1)

    elif num_point == 3:
        tree = scipy.spatial.cKDTree(gt_points, leafsize=2048)
        distances, locations = tree.query(gt_points, k=num_point)
        distances = np.delete(distances, 0, axis=1)
        distances = (distances[:, 0] + distances[:, 1]) / 2
        distances = torch.from_numpy(distances).unsqueeze(1)

    return distances

def reshape_train_data(image: Tensor, label: List[Tensor]) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    labels = []
    for points in label:
        kpoint = np.zeros((image.shape[2], image.shape[3]), dtype=np.uint8)
        for i in range(0, len(points)):
            if int(points[i][1]) < image.shape[2] and int(points[i][0]) < image.shape[3]:
                kpoint[int(points[i][1]), int(points[i][0])] = 1
        num_points = int(np.sum(kpoint))
        gt_points = np.nonzero(torch.from_numpy(kpoint))
        distances = calculate_knn_distance(gt_points, num_points, num_knn=4)
        points = torch.cat([gt_points, distances], dim=1)

        target = {}
        target['labels'] = torch.ones([1, num_points]).squeeze(0).type(torch.LongTensor).to(image.device)
        target['points_matcher'] = torch.true_divide(points, image.shape[3]).type(torch.FloatTensor).to(image.device)
        target['points'] = torch.true_divide(points[:, 0:3], image.shape[3]).type(torch.FloatTensor).to(image.device)
        labels.append(target)
    return image, labels

def reshape_eval_data(image: Tensor, label: List[Tensor], crop_size: int) -> Tuple[Tensor, Tensor]:
    points = [x.tolist() for y in label for x in y]
    kpoint = np.zeros((image.shape[2], image.shape[3]))
    for i in range(0, len(points)):
        if int(points[i][1]) < image.shape[2] and int(points[i][0]) < image.shape[3]:
            kpoint[int(points[i][1]), int(points[i][0])] = 1

    kpoint = torch.from_numpy(kpoint)
    padding_h = image.shape[2] % crop_size
    padding_w = image.shape[3] % crop_size

    if padding_w != 0:
        padding_w = crop_size - padding_w
    if padding_h != 0:
        padding_h = crop_size - padding_h

    pd = (padding_w, 0, padding_h, 0)
    image = F.pad(image, pd, 'constant')
    kpoint = F.pad(kpoint, pd, 'constant').unsqueeze(0)

    width, height = image.shape[3], image.shape[2]
    num_w = int(width / crop_size)
    num_h = int(height / crop_size)
    image = image.view(3, num_h, crop_size, width).view(3, num_h, crop_size, num_w, crop_size)
    image = image.permute(0, 1, 3, 2, 4).contiguous().view(3, num_w * num_h, crop_size, crop_size).permute(1, 0, 2, 3)
    kpoint = kpoint.view(num_h, crop_size, width).view(num_h, crop_size, num_w, crop_size)
    kpoint = kpoint.permute(0, 2, 1, 3).contiguous().view(num_w * num_h, 1, crop_size, crop_size)
    return image, kpoint


def get_points(
    original_image: np.ndarray,
    out_logits: Tensor,
    out_point: Tensor,
    crop_size: int,
    num_queries: int
) -> List[List[int]]:
    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_queries, dim=1)
    topk_points = topk_indexes // out_logits.shape[2]
    out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
    out_point = out_point * crop_size
    out_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)

    kpoint_list = []
    height = original_image.shape[0]
    width = original_image.shape[1]
    num_h = math.ceil(height / crop_size)
    num_w = math.ceil(width / crop_size)
    resized_height = num_h * crop_size
    resized_width = num_w * crop_size
    for i in range(len(out_points)):
        out_value = out_points[i].squeeze(0)[:, 0].data.cpu().numpy()
        out_point = out_points[i].squeeze(0)[:, 1:3].data.cpu().numpy().tolist()
        k = np.zeros((crop_size, crop_size))

        for j in range(len(out_point)):
            if out_value[j] < 0.25:
                break
            x = int(out_point[j][0])
            y = int(out_point[j][1])
            k[x, y] = 1

        kpoint_list.append(k)

    kpoint = torch.from_numpy(np.array(kpoint_list)).unsqueeze(0)
    kpoint = kpoint.view(num_h, num_w, crop_size, crop_size).permute(0, 2, 1, 3).contiguous()
    kpoint = kpoint.view(num_h, crop_size, resized_width).view(resized_height, resized_width).cpu().numpy()
    kpoint = cv2.resize(kpoint, (width, height))
    pred_points = np.nonzero(kpoint)
    pred_points = [[y, x] for x, y in zip(*pred_points)]
    return pred_points