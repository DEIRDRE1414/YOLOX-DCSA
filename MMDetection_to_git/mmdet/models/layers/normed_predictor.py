# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS

MODELS.register_module('Linear', module=nn.Linear)


@MODELS.register_module(name='NormedLinear')
class NormedLinear(nn.Linear):
    """Normalized Linear Layer.

    Args:
        tempeature (float, optional): Tempeature term. Defaults to 20.
        power (int, optional): Power term. Defaults to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Defaults to 1e-6.
    """

    def __init__(self,
                 *args,
                 tempearture: float = 20,
                 power: int = 1.0,
                 eps: float = 1e-6,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.eps = eps
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights."""
        nn.init.normal_(self.weight, mean=0, std=0.01)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for `NormedLinear`."""
        weight_ = self.weight / (
            self.weight.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture

        return F.linear(x_, weight_, self.bias)


@MODELS.register_module(name='NormedConv2d')
class NormedConv2d(nn.Conv2d):
    """Normalized Conv2d Layer.

    Args:
        tempeature (float, optional): Tempeature term. Defaults to 20.
        power (int, optional): Power term. Defaults to 1.0.
        eps (float, optional): The minimal value of divisor to
             keep numerical stability. Defaults to 1e-6.
        norm_over_kernel (bool, optional): Normalize over kernel.
             Defaults to False.
    """

    def __init__(self,
                 *args,
                 tempearture: float = 20,
                 power: int = 1.0,
                 eps: float = 1e-6,
                 norm_over_kernel: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.norm_over_kernel = norm_over_kernel
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for `NormedConv2d`."""
        if not self.norm_over_kernel:
            weight_ = self.weight / (
                self.weight.norm(dim=1, keepdim=True).pow(self.power) +
                self.eps)
        else:
            weight_ = self.weight / (
                self.weight.view(self.weight.size(0), -1).norm(
                    dim=1, keepdim=True).pow(self.power)[..., None, None] +
                self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture

        if hasattr(self, 'conv2d_forward'):
            x_ = self.conv2d_forward(x_, weight_)
        else:
            if torch.__version__ >= '1.8':
                x_ = self._conv_forward(x_, weight_, self.bias)
            else:
                x_ = self._conv_forward(x_, weight_)
        return x_

@MODELS.register_module(name='SahiConv2d')
class SahiConv2d(nn.Conv2d):
    def __init__(self, in_channels=[128, 256, 512], out_channels=128, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SahiConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # 生成随机的sahi mask，大小和输入张量一样
        sahi_mask = torch.randint_like(x, high=2)
        sahi_mask = sahi_mask.type(torch.float32)

        # 将sahi mask应用到输入张量中
        x = x * sahi_mask

        # 使用卷积层对sahi操作后的输入张量进行卷积
        x = self.conv(x)

        return x

    def __init__(self,
                 *args,
                 tempearture: float = 20,
                 power: int = 1.0,
                 eps: float = 1e-6,
                 norm_over_kernel: bool = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tempearture = tempearture
        self.power = power
        self.norm_over_kernel = norm_over_kernel
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for `NormedConv2d`."""
        if not self.norm_over_kernel:
            weight_ = self.weight / (
                self.weight.norm(dim=1, keepdim=True).pow(self.power) +
                self.eps)
        else:
            weight_ = self.weight / (
                self.weight.view(self.weight.size(0), -1).norm(
                    dim=1, keepdim=True).pow(self.power)[..., None, None] +
                self.eps)
        x_ = x / (x.norm(dim=1, keepdim=True).pow(self.power) + self.eps)
        x_ = x_ * self.tempearture

        if hasattr(self, 'conv2d_forward'):
            x_ = self.conv2d_forward(x_, weight_)
        else:
            if torch.__version__ >= '1.8':
                x_ = self._conv_forward(x_, weight_, self.bias)
            else:
                x_ = self._conv_forward(x_, weight_)
        return x_
