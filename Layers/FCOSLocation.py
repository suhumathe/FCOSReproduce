# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:30:05 2020

@author: sks
"""
import torch
import torch.nn as nn
from cfgs.config import cfg

class FCOSLocation(nn.Module):
    def __init__(self):
        super(FCOSLocation, self).__init__()
        self.fpn_stride = cfg.FCOS.FPN_STRIDES

    def forward(self, feature_shapes):
        """
        calculates the center of the receptive field of the feature points
        feature_shapes: a list,
        each element indicating the size of the feature map on the level [5, (B, C, H, W)]
        return a list, each element corresponds to a feature in the features
        """
        locations = []
        for level, feature_shape in enumerate(feature_shapes):
            h, w = feature_shape[-2:]
            stride = self.fpn_strides[level]
            shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32)
            shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32)
            shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)
            shifts_x = shifts_x.view(-1)
            shifts_y = shifts_y.view(-1)
            locations_per_level = torch.stack((shifts_x, shifts_y), dim=1) + stride // 2
            locations.append(locations_per_level)
        return locations
