# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Lang Huang, RainbowSecret from:
#   https://github.com/openseg-group/openseg.pytorch/blob/master/lib/models/modules/isa_block.py
# --------------------------------------------------------

import math
import warnings
from typing import Optional, Any, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import to_2tuple, trunc_normal_
from torch import nn, Tensor
from torch.nn.functional import linear, pad, softmax, dropout
from torch.overrides import has_torch_function, handle_torch_function

from ..modules.multihead_attention import MultiheadAttention, MultiheadAttentionRPE


class PadBlock(object):
    def __init__(
        self, 
        local_group_size: Union[int, Tuple[int, int]] = 7
    ) -> None:
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def pad_if_needed(
        self, 
        x: Tensor, 
        size: Tuple[int, int, int, int]
    ) -> Tensor:
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
            return F.pad(
                x,
                (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return x

    def depad_if_needed(
        self, 
        x: Tensor, 
        size: Tuple[int, int, int, int]
    ) -> Tensor:
        n, h, w, c = size
        pad_h = math.ceil(h / self.lgs[0]) * self.lgs[0] - h
        pad_w = math.ceil(w / self.lgs[1]) * self.lgs[1] - w
        if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
            return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :]
        return x


class LocalPermuteModule(object):
    def __init__(
        self, 
        local_group_size: Union[int, Tuple[int, int]] = 7
    ) -> None:
        self.lgs = local_group_size
        if not isinstance(self.lgs, (tuple, list)):
            self.lgs = to_2tuple(self.lgs)
        assert len(self.lgs) == 2

    def permute(
        self, 
        x: Tensor, 
        size: Tuple[int, int, int, int]
    ) -> Tensor:
        n, h, w, c = size
        return rearrange(
            x,
            "n (qh ph) (qw pw) c -> (ph pw) (n qh qw) c",
            n=n,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )

    def rev_permute(
        self, 
        x: Tensor, 
        size: Tuple[int, int, int, int]
    ) -> Tensor:
        n, h, w, c = size
        return rearrange(
            x,
            "(ph pw) (n qh qw) c -> n (qh ph) (qw pw) c",
            n=n,
            qh=h // self.lgs[0],
            ph=self.lgs[0],
            qw=w // self.lgs[0],
            pw=self.lgs[0],
            c=c,
        )


class MHA_(MultiheadAttention):
    def __init__(
        self, 
        *args: Any,
        rpe: bool = False,
        window_size: int = 7,
        **kwargs: Any
    ) -> None:
        super(MHA_, self).__init__(*args, **kwargs)

        self.rpe = rpe
        if rpe:
            self.window_size = [window_size] * 2
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                    self.num_heads,
                )
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self, 
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        do_qkv_proj: bool = True,
        do_out_proj: bool = True,
        rpe: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not self._qkv_same_embed_dim:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )
        else:
            return self.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                out_dim=self.vdim,
                do_qkv_proj=do_qkv_proj,
                do_out_proj=do_out_proj,
                rpe=rpe,
            )

    def multi_head_attention_forward(
        self, 
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        out_dim: Optional[int] = None,
        do_qkv_proj: bool = True,
        do_out_proj: bool = True,
        rpe: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not torch.jit.is_scripting():
            tens_ops = (
                query,
                key,
                value,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                out_proj_weight,
                out_proj_bias,
            )
            if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(
                tens_ops
            ):
                return handle_torch_function(
                    multi_head_attention_forward,
                    tens_ops,
                    query,
                    key,
                    value,
                    embed_dim_to_check,
                    num_heads,
                    in_proj_weight,
                    in_proj_bias,
                    bias_k,
                    bias_v,
                    add_zero_attn,
                    dropout_p,
                    out_proj_weight,
                    out_proj_bias,
                    training=training,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=use_separate_proj_weight,
                    q_proj_weight=q_proj_weight,
                    k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight,
                    static_k=static_k,
                    static_v=static_v,
                )
        tgt_len, bsz, embed_dim = query.size()
        key = query if key is None else key
        value = query if value is None else value

        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        v_head_dim = out_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.q_proj(query) * scaling if do_qkv_proj else query
        k = self.k_proj(key) if do_qkv_proj else key
        v = self.v_proj(value) if do_qkv_proj else value

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )

        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, v_head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if self.rpe and rpe:
            assert (
                src_len == self.window_size[0] * self.window_size[1]
                and tgt_len == self.window_size[0] * self.window_size[1]
            ), f"src{src_len}, tgt{tgt_len}, window{self.window_size[0]}"
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            ) + relative_position_bias.unsqueeze(0)
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, v_head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, out_dim)
        )
        if do_out_proj:
            attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, q, k, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, q, k


class MultiheadISAAttention(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        num_heads: int,
        window_size: int = 7,
        attn_type: str = "isa_local",
        rpe: bool = True,
        **kwargs: Any
    ) -> None:
        super(MultiheadISAAttention, self).__init__()

        self.dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_type = attn_type
        self.with_rpe = rpe

        self.attn = MultiheadAttentionRPE(
            embed_dim, num_heads, rpe=rpe, window_size=window_size, **kwargs
        )
        self.pad_helper = PadBlock(window_size)
        assert attn_type in ["isa_local"]
        if attn_type == "isa_local":
            self.permute_helper = LocalPermuteModule(window_size)
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")

    def forward(
        self, 
        x: Tensor, 
        H: int, 
        W: int, 
        **kwargs: Any
    ) -> Tensor:
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        if self.attn_type in ["isa_local"]:
            x_pad = self.pad_helper.pad_if_needed(x, x.size())
            x_permute = self.permute_helper.permute(x_pad, x_pad.size())
            out, _, _ = self.attn(
                x_permute, x_permute, x_permute, rpe=self.with_rpe, **kwargs
            )
            out = self.permute_helper.rev_permute(out, x_pad.size())
        else:
            raise NotImplementedError("We only support ['isa_local'] Now.")
        out = self.pad_helper.depad_if_needed(out, x.size())
        return out.reshape(B, N, C)

    def extra_repr(
        self
    ) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(
        self, 
        N: int
    ) -> int:
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops
