from typing import List, Dict

from torch import nn, Tensor


class APGCC(nn.Module):
    def __init__(
        self, 
        config: object,
        sync_bn: bool = False,
        last_pool: bool = False
    ) -> None:
        super().__init__()
        self.backbone = config.backbone.lower()
        self.encoder = self.build_encoder(self.backbone, last_pool)
        self.decoder = self.build_decoder(decoder_type=config.decoder_type,
                                         in_planes=self.encoder.get_outplanes(),
                                         line=config.line,
                                         row=config.row,
                                         num_anchor_points=config.line * config.row,
                                         sync_bn=sync_bn,
                                         aux_enabled=config.aux_enabled,
                                         aux_num_layers=config.aux_num_layers,
                                         aux_range=config.aux_range,
                                         aux_kwargs={
                                             'pos_coef': config.pos_coef,
                                             'neg_coef': config.neg_coef,
                                             'pos_loc': config.pos_loc,
                                             'neg_loc': config.neg_loc
                                         })

    def build_encoder(self, backbone: str, last_pool: bool) -> nn.Module:
        if backbone in ['vgg16', 'vgg16_bn']:
            from .modules import Base_VGG as build_encoder
        elif backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            from .modules import Base_ResNet as build_encoder
        else:
            raise ValueError("Select appropriate backbone. Available backbones are (vgg16, vgg16bn, resnet18, resnet34, resnet50, resnet101)")
        return build_encoder(name=backbone, last_pool=last_pool)

    def build_decoder(
        self, 
        decoder_type: str,
        in_planes: List[int],
        line: int,
        row: int,
        num_anchor_points: int,
        sync_bn: bool,
        aux_enabled: bool,
        aux_num_layers: List[int],
        aux_range: List[int],
        aux_kwargs: Dict[str, float]
    ) -> nn.Module:
        if decoder_type == 'basic':
            from .modules import Basic_Decoder_Model as build_decoder
        elif decoder_type == 'IFI':
            from .modules import IFI_Decoder_Model as build_decoder
        decoder = build_decoder(in_planes=in_planes,
                                num_classes=2,
                                num_anchor_points=num_anchor_points,
                                line=line,
                                row=row,
                                sync_bn=sync_bn,
                                aux_enabled=aux_enabled,
                                aux_num_layers=aux_num_layers,
                                aux_range=aux_range,
                                aux_kwargs=aux_kwargs)
        return decoder

    def forward(self, x: Tensor) -> Dict:
        features = self.encoder(x)
        out = self.decoder(x, features)
        return out


