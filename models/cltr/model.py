import math
from typing import Dict, List

import torch
from core.utils import NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid
from torch import nn, Tensor

from .modules import build_backbone, Transformer, MLP


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """
    def __init__(
        self, 
        backbone_model: nn.Module,
        transformer_model: nn.Module,
        num_classes: int,
        num_queries: int,
        channel_point: int,
        auxiliary_loss: bool = False,
    ) -> None:
        """ Initializes the model.
        Parameters:
            backbone_model: torch module of the backbone to be used. See backbone.py
            transformer_model: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            auxiliary_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone_model = backbone_model
        self.num_queries = num_queries
        self.transformer_model = transformer_model
        hidden_dim = self.transformer_model.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.point_embed = MLP(hidden_dim, hidden_dim, channel_point, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone_model.num_channels, hidden_dim, kernel_size=1)
        self.auxiliary_loss = auxiliary_loss

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init point_mebed
        nn.init.constant_(self.point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].bias.data, 0)

    def forward(self, x: NestedTensor) -> Dict[str, Tensor]:
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_points": The normalized points coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(x, (list, torch.Tensor)):
            x = nested_tensor_from_tensor_list(x)
        features, pos = self.backbone_model(x)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs, reference = self.transformer_model(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        reference_before_sigmoid = inverse_sigmoid(reference)
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            tmp = self.point_embed(hs[lvl])
            tmp[..., :2] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)

        outputs_class = self.class_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_points': outputs_coord[-1]}
        if self.auxiliary_loss:
            out['aux_outputs'] = self._set_auxiliary_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_auxiliary_loss(self, outputs_class: Tensor, outputs_coord: Tensor) -> List[Dict[str, Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_points': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

def CLTR(config: object) -> ConditionalDETR:
    backbone_model = build_backbone(
        backbone=config.backbone,
        hidden_dim=config.hidden_dim,
        position_embedding=config.position_embedding,
        lr_backbone=config.lr_backbone,
        dilation=config.dilation
    )

    transformer_model = Transformer(
        d_model=config.hidden_dim,
        dropout=config.dropout,
        num_heads=config.num_heads,
        num_queries=config.num_queries,
        dim_feedforward=config.dim_feedforward,
        num_encoder_layers=config.encoder_layers,
        num_decoder_layers=config.decoder_layers,
        normalize_before=config.pre_norm,
        return_intermediate_dec=True,
    )

    return ConditionalDETR(
        backbone_model,
        transformer_model,
        num_classes=2,
        num_queries=config.num_queries,
        channel_point=config.channel_point,
        auxiliary_loss=config.auxiliary_loss,
    )







