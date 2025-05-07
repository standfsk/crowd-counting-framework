import math
from functools import partial
from typing import List, Optional, Callable, Tuple
from typing import Union, Any

import torch
import torch.nn.functional as F
from losses import dmloss, daceloss
from torch import nn, Tensor

num_to_word = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen",
    "20": "twenty", "21": "twenty-one", "22": "twenty-two", "23": "twenty-three", "24": "twenty-four", "25": "twenty-five", "26": "twenty-six", "27": "twenty-seven", "28": "twenty-eight", "29": "twenty-nine",
    "30": "thirty", "31": "thirty-one", "32": "thirty-two", "33": "thirty-three", "34": "thirty-four", "35": "thirty-five", "36": "thirty-six", "37": "thirty-seven", "38": "thirty-eight", "39": "thirty-nine",
    "40": "forty", "41": "forty-one", "42": "forty-two", "43": "forty-three", "44": "forty-four", "45": "forty-five", "46": "forty-six", "47": "forty-seven", "48": "forty-eight", "49": "forty-nine",
    "50": "fifty", "51": "fifty-one", "52": "fifty-two", "53": "fifty-three", "54": "fifty-four", "55": "fifty-five", "56": "fifty-six", "57": "fifty-seven", "58": "fifty-eight", "59": "fifty-nine",
    "60": "sixty", "61": "sixty-one", "62": "sixty-two", "63": "sixty-three", "64": "sixty-four", "65": "sixty-five", "66": "sixty-six", "67": "sixty-seven", "68": "sixty-eight", "69": "sixty-nine",
    "70": "seventy", "71": "seventy-one", "72": "seventy-two", "73": "seventy-three", "74": "seventy-four", "75": "seventy-five", "76": "seventy-six", "77": "seventy-seven", "78": "seventy-eight", "79": "seventy-nine",
    "80": "eighty", "81": "eighty-one", "82": "eighty-two", "83": "eighty-three", "84": "eighty-four", "85": "eighty-five", "86": "eighty-six", "87": "eighty-seven", "88": "eighty-eight", "89": "eighty-nine",
    "90": "ninety", "91": "ninety-one", "92": "ninety-two", "93": "ninety-three", "94": "ninety-four", "95": "ninety-five", "96": "ninety-six", "97": "ninety-seven", "98": "ninety-eight", "99": "ninety-nine",
    "100": "one hundred", "200": "two hundred", "300": "three hundred", "400": "four hundred", "500": "five hundred", "600": "six hundred", "700": "seven hundred", "800": "eight hundred", "900": "nine hundred",
    "1000": "one thousand"
}


def num2word(num: Union[int, str]) -> str:
    """
    Convert the input number to the corresponding English word. For example, 1 -> "one", 2 -> "two", etc.
    """
    num = str(int(num))
    return num_to_word.get(num, num)


def format_count(count: Union[float, Tuple[float, float]], prompt_type: str = "word") -> str:
    if count == 0:
        return "There is no person." if prompt_type == "word" else "There is 0 person."
    elif count == 1:
        return "There is one person." if prompt_type == "word" else "There is 1 person."
    elif isinstance(count, (int, float)):
        return f"There are {num2word(int(count))} people." if prompt_type == "word" else f"There are {int(count)} people."
    elif count[1] == float("inf"):
        return f"There are more than {num2word(int(count[0]))} people." if prompt_type == "word" else f"There are more than {int(count[0])} people."
    else:  # count is a tuple of finite numbers
        left, right = int(count[0]), int(count[1])
        left, right = num2word(left), num2word(right) if prompt_type == "word" else left, right
        return f"There are between {left} and {right} people."

def get_loss_fn(config: object) -> nn.Module:
    if config.bins is None:
        assert config.weight_ot is not None and config.weight_tv is not None, f"Expected weight_ot and weight_tv to be not None, got {args.weight_ot} and {args.weight_tv}"
        loss_fn = dmloss(
            input_size=config.input_size,
            reduction=config.reduction,
        )
    else:
        loss_fn = daceloss(
            bins=config.bins,
            reduction=config.reduction,
            weight_count_loss=config.weight_count_loss,
            count_loss_type=config.count_loss_type,
            input_size=config.input_size,
        )
    return loss_fn


def cosine_annealing_warm_restarts(
    epoch: int,
    base_lr: float,
    warmup_epochs: int,
    warmup_lr: float,
    T_0: int,
    T_mult: int,
    eta_min: float,
) -> float:
    """
    Learning rate scheduler.
    The learning rate will linearly increase from warmup_lr to lr in the first warmup_epochs epochs.
    Then, the learning rate will follow the cosine annealing with warm restarts strategy.
    """
    assert epoch >= 0, f"epoch must be non-negative, got {epoch}."
    assert isinstance(warmup_epochs,
                      int) and warmup_epochs >= 0, f"warmup_epochs must be non-negative, got {warmup_epochs}."
    assert isinstance(warmup_lr, float) and warmup_lr > 0, f"warmup_lr must be positive, got {warmup_lr}."
    assert isinstance(T_0, int) and T_0 >= 1, f"T_0 must be greater than or equal to 1, got {T_0}."
    assert isinstance(T_mult, int) and T_mult >= 1, f"T_mult must be greater than or equal to 1, got {T_mult}."
    assert isinstance(eta_min, float) and eta_min > 0, f"eta_min must be positive, got {eta_min}."
    assert isinstance(base_lr, float) and base_lr > 0, f"base_lr must be positive, got {base_lr}."
    assert base_lr > eta_min, f"base_lr must be greater than eta_min, got base_lr={base_lr} and eta_min={eta_min}."
    assert warmup_lr >= eta_min, f"warmup_lr must be greater than or equal to eta_min, got warmup_lr={warmup_lr} and eta_min={eta_min}."

    if epoch < warmup_epochs:
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
    else:
        epoch -= warmup_epochs
        if T_mult == 1:
            T_cur = epoch % T_0
            T_i = T_0
        else:
            n = int(math.log((epoch / T_0 * (T_mult - 1) + 1), T_mult))
            T_cur = epoch - T_0 * (T_mult ** n - 1) / (T_mult - 1)
            T_i = T_0 * T_mult ** (n)

        lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2

    return lr / base_lr

def get_optimizer(config: object, model: nn.Module) -> Tuple:
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=partial(
            cosine_annealing_warm_restarts,
            warmup_epochs=config.warmup_epochs,
            warmup_lr=config.warmup_lr,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            base_lr=config.lr
        ),
    )

    return optimizer, scheduler

def conv3x3(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.stride = stride
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        expansion: int = 4,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        self.expansion = expansion
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out

def _init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

def make_resnet_layers(
    block: Union[BasicBlock, Bottleneck],
    cfg: List[Union[int, str]],
    in_channels: int,
    dilation: int = 1,
    expansion: int = 1,
) -> nn.Sequential:
    layers = []
    for v in cfg:
        if v == "U":
            layers.append(F.interpolate(v, scale_factor=2, mode="bilinear"))
        else:
            layers.append(block(
                in_channels=in_channels,
                out_channels=v,
                dilation=dilation,
                expansion=expansion,
            ))
            in_channels = v

    layers = nn.Sequential(*layers)
    layers.apply(_init_weights)
    return layers