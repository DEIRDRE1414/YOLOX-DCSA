# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .se_layer import ChannelAttention


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.暗网中使用的基本瓶颈块

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.每个ResBlock由两个ConvModule组成，输入被添加到最终输出。每个ConvModule由Conv、BN和LeakyReLU组成。第一个 convLayer 的过滤器大小为 1x1，第二个 convLayer 的过滤器大小为 3x3

    Args:
        in_channels (int): The input channels of this Module.该模块的输入通道
        out_channels (int): The output channels of this Module.该模块的输出通道
        expansion (float): The kernel size of the convolution.卷积核大小
            Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out.是否在out中添加身份
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.是否使用深度可分离卷积
            Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.卷积层的配置字典。默认为 None，这意味着使用 conv2d
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').规范化层的配置字典
        act_cfg (dict): Config dict for activation layer.激活层的配置字典
            Defaults to dict(type='Swish').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPNeXtBlock(BaseModule):
    """The basic bottleneck block used in CSPNeXt.CSPNeXt 中使用的基本瓶颈块

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): Expand ratio of the hidden channel. Defaults to 0.5.扩展（float）：隐藏通道的扩展比例。默认为 0.5。
        add_identity (bool): Whether to add identity to the out. Only works
            when in_channels == out_channels. Defaults to True.扩展（float）：隐藏通道的扩展比例。默认为 0.5。
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False. 是否使用深度可分离卷积。
        kernel_size (int): The kernel size of the second convolution layer.
            Defaults to 5.第二个卷积层的内核大小。
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d. 卷积层的配置字典。默认为 None，这意味着使用 conv2d
        norm_cfg (dict): Config dict for normalization layer.规范化层的配置字典
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 kernel_size: int = 5,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPLayer(BaseModule):
    """Cross Stage Partial Layer.跨阶段部分层

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. 调整隐藏层通道数的比率Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1.
        add_identity (bool): Whether to add identity in blocks.是否在块中添加标识
            Defaults to True.
        use_cspnext_block (bool): Whether to use CSPNeXt block.是否使用CSPNeXt块
            Defaults to False.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks.是否在块中使用深度可分离卷积 Defaults to False.
        channel_attention (bool): Whether to add channel attention in each
            stage.是否在每个阶段增加渠道关注度 Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish')
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 1,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 use_cspnext_block: bool = False,
                 channel_attention: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            block(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:

        #

            x_final = self.attention(x_final)




        return self.final_conv(x_final)

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