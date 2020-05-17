# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:31:40 2020

@author: sks
"""
import torch
from torch import nn

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

def conv_with_kaiming_uniform(in_channels, out_channels, kernel_size, stride=1, dilation=1):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=dilation * (kernel_size - 1) // 2,
        dilation=dilation,
        bias=True
    )
    nn.init.kaiming_uniform_(conv.weight, a=1)
    nn.init.constant_(conv.bias, 0)
    module = [conv,]
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv
