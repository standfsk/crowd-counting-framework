from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .utils import Gaussianlayer


class InheritanceLoss(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.counter_type = config.counter_type
        self.weight = config.density_factor
        self.resolution_num = config.resolution_num
        self.baseline_loss = config.baseline_loss
        self.loss_weight = config.loss_weight

        self.route_size = [config.route_size // (2 ** self.resolution_num[0])] * 2
        self.label_start = self.resolution_num[0]
        self.label_end = self.resolution_num[-1] + 1
        
        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs: List[Tensor], targets: List[Tensor]) -> Dict:
        if self.counter_type == "single_resolution":
            targets = targets[0].unsqueeze(1)
            targets = self.gaussian(targets)

            loss = self.mse_loss(outputs, targets * self.weight)
            gt_cnt = targets.sum().item()
            pred_cnt = outputs.sum().item() / self.weight

            result = {
                'x4': {'gt': gt_cnt, 'error': max(0, gt_cnt - abs(gt_cnt - pred_cnt))},
                'x8': {'gt': 0, 'error': 0},
                'x16': {'gt': 0, 'error': 0},
                'x32': {'gt': 0, 'error': 0},
                'acc1': {'gt': 0, 'error': 0},
                'loss': loss,
                'pred_den':
                    {
                        '1': outputs / self.weight,
                    },
                'gt_den': {'1': targets}
            }
            return result

        elif self.counter_type == "withMOE":
            result = {'pred_den': {}, 'gt_den': {}}
            targets_list = []
            targets = targets[self.label_start:self.label_end]
            for i, target in enumerate(targets):
                targets_list.append(self.gaussian(target.unsqueeze(1)) * self.weight)

            moe_label, score_gt = self.get_moe_label(outputs, targets_list, self.route_size)
            mask_gt = torch.zeros_like(score_gt)
            mask_gt = mask_gt.scatter_(1, moe_label, 1)

            loss_list = []
            out = torch.zeros_like(outputs[0])
            target_patch = torch.zeros_like(targets_list[0])
            result.update({'acc1': {'gt': 0, 'error': 0}})

            mask_add = torch.ones_like(mask_gt[:, 0].unsqueeze(1))
            for i in range(mask_gt.size(1)):
                kernel = (int(self.route_size[0] / (2 ** i)), int(self.route_size[1] / (2 ** i)))
                loss_mask = F.interpolate(mask_add, size=(outputs[i].size()[2:]), mode="nearest")
                hard_loss = self.mse_loss(outputs[i] * loss_mask, targets_list[i] * loss_mask)
                loss_list.append(hard_loss)

                if i == 0:
                    target_patch += (targets_list[0] * F.interpolate(mask_gt[:, i].unsqueeze(1), size=(outputs[i].size()[2:]), mode="nearest"))
                    target_patch = F.unfold(target_patch, kernel, stride=kernel)
                    B_, _, L_ = target_patch.size()
                    target_patch = target_patch.transpose(2, 1).view(B_, L_, kernel[0], kernel[1])
                else:
                    gt_slice = F.unfold(targets_list[i], kernel, stride=kernel)
                    B, KK, L = gt_slice.size()

                    pick_gt_idx = (moe_label.flatten(start_dim=1) == i).unsqueeze(2).unsqueeze(3)
                    gt_slice = gt_slice.transpose(2, 1).view(B, L, kernel[0], kernel[1])
                    pad_w, pad_h = (self.route_size[1] - kernel[1]) // 2, (self.route_size[0] - kernel[0]) // 2
                    gt_slice = F.pad(gt_slice, [pad_w, pad_w, pad_h, pad_h], "constant", 0.2)
                    gt_slice = (gt_slice * pick_gt_idx)
                    target_patch += gt_slice

                gt_cnt = (targets_list[i] * loss_mask).sum().item() / self.weight
                pred_cnt = (outputs[i] * loss_mask).sum().item() / self.weight
                result.update({f"x{2 ** (self.resolution_num[i] + 2)}": {'gt': gt_cnt, 'error': max(0, gt_cnt - abs(gt_cnt - pred_cnt))}})
                mask_add -= mask_gt[:, i].unsqueeze(1)

            B_num, C_num, H_num, W_num = outputs[0].size()
            patch_h, patch_w = H_num // self.route_size[0], W_num // self.route_size[1]
            target_patch = target_patch.view(B_num, patch_h * patch_w, -1).transpose(1, 2)
            target_patch = F.fold(target_patch, output_size=(H_num, W_num), kernel_size=self.route_size, stride=self.route_size)

            loss = 0
            if self.baseline_loss:
                loss = loss_list[0]
            else:
                for i in range(len(self.resolution_num)):
                    loss += loss_list[i] * self.loss_weight[i]

            for i in ['x4', 'x8', 'x16', 'x32']:
                if i not in result.keys():
                    result.update({i: {'gt': 0, 'error': 0}})
            result.update({'moe_label': moe_label})
            result.update({'loss': torch.unsqueeze(loss, 0)})
            result['pred_den'].update({'1': outputs[0] / self.weight})
            result['pred_den'].update({'8': outputs[-1] / self.weight})
            result['gt_den'].update({'1': target_patch / self.weight})
            result['gt_den'].update({'8': targets_list[-1] / self.weight})

            return result
        else:
            raise ValueError("Wrong counter type selected")

    def get_moe_label(
        self, 
        outputs: List[Tensor],
        targets_list: List[Tensor],
        route_size: List[int]
    ) -> Tuple[Tensor, Tensor]:
        B_num, C_num, H_num, W_num = outputs[0].size()
        patch_h, patch_w = H_num // route_size[0], W_num // route_size[1]
        errorInslice_list = []

        for i, (pred, gt) in enumerate(zip(outputs, targets_list)):
            kernel = (int(route_size[0] / (2 ** i)), int(route_size[1] / (2 ** i)))

            weight = torch.full(kernel, 1 / (kernel[0] * kernel[1])).expand(1, pred.size(1), -1, -1)
            weight = nn.Parameter(data=weight, requires_grad=False).to(pred.device)

            error = (pred - gt) ** 2
            patch_mse = F.conv2d(error, weight, stride=kernel)

            weight = torch.full(kernel, 1.).expand(1, pred.size(1), -1, -1)
            weight = nn.Parameter(data=weight, requires_grad=False).to(pred.device)

            patch_error = F.conv2d(error, weight, stride=kernel)  # (pred-gt)*(gt>0)
            fractions = F.conv2d(gt, weight, stride=kernel)
            instance_mse = patch_error / (fractions + 1e-10)
            errorInslice_list.append(patch_mse + instance_mse)

        score = torch.cat(errorInslice_list, dim=1)
        moe_label = score.argmin(dim=1, keepdim=True)
        return moe_label, score


def steerer_loss(config: object) -> InheritanceLoss:
    criterion = InheritanceLoss(config)
    return criterion