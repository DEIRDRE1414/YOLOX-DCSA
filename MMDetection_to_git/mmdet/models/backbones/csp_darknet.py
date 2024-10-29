# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..layers import CSPLayer


class SahiConv2d(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
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
class Focus(nn.Module):
    """Focus width and height information into channel space.将宽度和高度信息聚焦到通道空间中

    Args:
        in_channels (int): The input channels of this Module.该模块的输入通道。
        out_channels (int): The output channels of this Module.该模块的输出通道
        kernel_size (int): The kernel size of the convolution. Default: 1.卷积核大小
        stride (int): The stride of the convolution. Default: 1.卷积的步长
        conv_cfg (dict): Config dict for convolution layer. Default: None,卷积层的配置字典
            which means using conv2d.这意味着使用 conv2d
        norm_cfg (dict): Config dict for normalization layer.规范化层的配置字典
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.激活层的配置字典
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.YOLOv3-SPP 中使用的空间金字塔池化层。

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling池化内核大小的顺序
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.卷积层的配置字典
        norm_cfg (dict): Config dict for normalization layer.规范化层的配置字典
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvModule(
            conv2_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@MODELS.register_module()
class CSPDarknet(BaseModule):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.CSP-Darknet 的架构
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.深度乘数，将 CSP 层中的块数乘以该数量。
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.宽度乘数，将每层中的通道数乘以该数量。
        out_indices (Sequence[int]): Output from which stages.从哪个阶段输出
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.要冻结的阶段（停止分级并设置评估模式）。 -1表示不冻结任何参数
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.是否使用深度可分离卷积。
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).覆盖默认的拱形设置。默认值：无。spp_kernal_sizes: (tuple[int]): SPP 层的内核大小的序列。默认值：(5,9,13)。
        conv_cfg (dict): Config dict for convolution layer. Default: None.卷积层的配置字典。默认值：无
        norm_cfg (dict): Dictionary to construct and config norm layer.用于构造和配置规范层的字典。
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.激活层的配置字典
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.是否将norm层设置为eval模式，即冻结运行统计数据（mean和var）。注意：仅对 Batch Norm 及其变体有影响
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.初始化配置字典
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # in_channels（输入通道数）：指定神经网络层或模型的输入张量的通道数。在卷积神经网络中，输入通道数表示输入特征图的深度。
    #
    # out_channels（输出通道数）：指定神经网络层或模型的输出张量的通道数。在卷积神经网络中，输出通道数表示输出特征图的深度。
    #
    # num_blocks（块数量）：表示神经网络中重复的块的数量。这些块可以是相同的层或模块的堆叠，用于增加网络的深度和复杂性。
    #
    # add_identity（添加身份连接）：在某些神经网络架构中，为了促进信息的流动和梯度传播，可以将输入张量直接添加到某些层的输出张量中。这种连接称为身份连接，可以帮助网络更好地学习和保留输入的信息。
    #
    # use_spp（使用空间金字塔池化）：空间金字塔池化（Spatial Pyramid Pooling，SPP）是一种在卷积神经网络中广泛使用的池化操作。
    # 它允许网络处理任意大小的输入图像，并生成固定大小的输出特征。通过在不同尺度上进行池化操作，SPP可以捕获输入图像中不同区域的上下文信息。
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        super().__init__(init_cfg)
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.stem = Focus(
            3,
            int(arch_setting[0][0] * widen_factor),
            kernel_size=3,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.layers = ['stem']

        for i, (in_channels, out_channels, num_blocks, add_identity,
                use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernal_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
