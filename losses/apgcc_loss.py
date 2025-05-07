from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from core.distributed import get_world_size, is_dist_avail_and_initialized
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor


class HungarianMatcher_Crowd(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> List[Tuple[Tensor, Tensor]]:
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                           objects in the target) containing the class labels
                 "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2] # num_queries = H/8*W/8*lines*row = feature_map*sample_points

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and points
        # tgt_ids = torch.cat([v["labels"] for v in targets])
        # tgt_idds = torch.cat([v["image_id"] for v in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets])   # torch.ones(gt.shape[0])
        tgt_points = torch.cat([v["point"] for v in targets]) # coords in gt map

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class.size = [bs*num_queries, num_all_gt_points]
        cost_class = -out_prob[:, tgt_ids]  # means confidence of pred, all_gt_points share same confidence.
        # cost_calss : out_prob(32*1028, 2) -> (32*1028, len(tgt_ids))

        # Compute the L2 cost between point
        # cost_point.size = [bs*num_queries, num_all_gt_points]
        cost_point = torch.cdist(out_points, tgt_points, p=2)
        # Compute the giou cost between point

        # Final cost matrix, default: w_cost_point = 0.05, w_cost_class=1
        C = self.cost_point * cost_point + self.cost_class * cost_class # compute each predict point to every gt(all batch) scores.

        # split to according patch predict. only want to consider within target points scores.
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["point"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]  # linear_sum_assignment find the best match.
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion_Crowd(nn.Module):
    # Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
    def __init__(
        self, 
        num_classes: int,
        matcher: HungarianMatcher_Crowd,
        weight_dict: Dict[str, float],
        eos_coef: float,
        aux_kwargs: Dict[str, Any]
    ) -> None:
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        if 'loss_aux' in self.weight_dict:
            self.aux_mode = False
        else:
            self.aux_mode = True
            self.aux_number = aux_kwargs['AUX_NUMBER']
            self.aux_range = aux_kwargs['AUX_RANGE']
            self.aux_kwargs = aux_kwargs['AUX_kwargs']

    def loss_labels(
        self, 
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_points: int
    ) -> Dict[str, Tensor]:
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # size=[batch*patch, num_queries, 2] [p0, p1]

        # make the gt array, only match idx set 1
        idx = self._get_src_permutation_idx(indices)  # batch_idx, src_idx
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # always 1 # size=[num_of_gt_point]
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o  # size=[batch*patch, num_queries] 0/1
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_points(
        self, 
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        indices: List[Tuple[Tensor, Tensor]],
        num_points: int
    ) -> Dict[str, float]:
        '''
        only compare to matched pairs
        '''
        assert 'pred_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.mse_loss(src_points, target_points, reduction='none')
        losses = {}
        losses['loss_points'] = loss_bbox.sum() / num_points
        return losses

    def loss_auxiliary(
        self, 
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        show: bool = False
    ) -> Dict[str, float]:
        # out: {"pred_logits", "pred_points", "offset"}
        # aux_out: {"pos0":out, "pos1":out, "neg0":out, "neg1":out, ...}
        loss_aux_pos = 0.
        loss_aux_neg = 0.
        loss_aux = 0.
        for n_pos in range(self.aux_number[0]):
            src_outputs = outputs['pos%d' % n_pos]
            # cls loss
            pred_logits = src_outputs['pred_logits']  # size=[1, # of gt anchors, 2] [p0, p1]
            target_classes = torch.ones(pred_logits.shape[:2], dtype=torch.int64,
                                        device=pred_logits.device)  # [1, # of gt anchors], all sample is the head class
            loss_ce_pos = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
            # loc loss
            pred_points = src_outputs['pred_points'][0]
            target_points = torch.cat([t['point'] for t in targets], dim=0)
            target_points = target_points.repeat(1, int(pred_points.shape[0] / target_points.shape[0]))
            target_points = target_points.reshape(-1, 2)
            loss_loc_pos = F.mse_loss(pred_points, target_points, reduction='none')
            loss_loc_pos = loss_loc_pos.sum() / pred_points.shape[0]
            loss_aux_pos += loss_ce_pos + self.aux_kwargs['pos_loc'] * loss_loc_pos
        loss_aux_pos /= (self.aux_number[0] + 1e-9)

        for n_neg in range(self.aux_number[1]):
            src_outputs = outputs['neg%d' % n_neg]
            # cls loss
            pred_logits = src_outputs['pred_logits']  # size=[1, # of gt anchors, 2] [p0, p1]
            target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.int64,
                                         device=pred_logits.device)  # [1, # of gt anchors], all sample is the head class
            loss_ce_neg = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
            # loc loss
            pred_points = src_outputs['offset'][0]
            target_points = torch.zeros(pred_points.shape, dtype=torch.float, device=pred_logits.device)
            loss_loc_neg = F.mse_loss(pred_points, target_points, reduction='none')
            loss_loc_neg = loss_loc_neg.sum() / pred_points.shape[0]
            loss_aux_neg += loss_ce_neg + self.aux_kwargs['neg_loc'] * loss_loc_neg
        loss_aux_neg /= (self.aux_number[1] + 1e-9)

        if show:
            if self.aux_number[0] > 0:
                print("Auxiliary Training: [Pos] loss_cls:", loss_ce_pos, " loss_loc:", loss_loc_pos, " loss:",
                      loss_aux_pos)
            if self.aux_number[1] > 0:
                print("Auxiliary Training: [Neg] loss_cls:", loss_ce_neg, " loss_loc:", loss_loc_neg, " loss:",
                      loss_aux_neg)
        loss_aux = self.aux_kwargs['pos_coef'] * loss_aux_pos + self.aux_kwargs['neg_coef'] * loss_aux_neg
        losses = {'loss_aux': loss_aux}
        return losses

    def _get_src_permutation_idx(self, indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        show: bool = False
    ) -> Dict[str, float]:
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {'pred_logits': outputs['pred_logits'], 'pred_points': outputs['pred_points'],
                   'offset': outputs['offset']}
        indices1 = self.matcher(output1,
                                targets)  # return (idx_of_pred, idx_of_gt). # pairs of indices # indices[batch] = (point_coords, gt_idx)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(output1.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        for loss in self.weight_dict.keys():
            if loss == 'loss_ce':
                losses.update(self.loss_labels(output1, targets, indices1, num_boxes))
            elif loss == 'loss_points':
                losses.update(self.loss_points(output1, targets, indices1, num_boxes))
            elif loss == 'loss_aux':
                out_auxs = output1['aux']
                losses.update(self.loss_auxiliary(out_auxs, targets, show))
            else:
                raise KeyError('do you really want to compute {} loss?'.format(loss))
        return losses

def apgcc_loss(config: object) -> SetCriterion_Crowd:
    weight_dict = {'loss_ce': 1., 'loss_points': 0.0002, 'loss_aux': 0.2}
    matcher = HungarianMatcher_Crowd(cost_class=config.set_cost_class, cost_point=config.set_cost_point)
    if not config.aux_enabled:
        del weight_dict['loss_aux']
    criterion = SetCriterion_Crowd(num_classes=1,
                                   matcher=matcher,
                                   weight_dict=weight_dict,
                                   eos_coef=config.eos_coef,
                                   aux_kwargs={'AUX_NUMBER': config.aux_num_layers,
                                               'AUX_RANGE': config.aux_range,
                                               'AUX_kwargs': {
                                                   'pos_coef': config.pos_coef,
                                                   'neg_coef': config.neg_coef,
                                                   'pos_loc': config.pos_loc,
                                                   'neg_loc': config.neg_loc
                                               }})
    return criterion

