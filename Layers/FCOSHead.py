# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:27:14 2020

@author: sks
"""

import math
import torch
from torch import nn
from cfgs.config import cfg

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class FCOSHead(nn.Module):

    def __init__(self, num_classes, in_channels):
        """
        num_classes: number of classes, i.e., 81 for coco
        in_channels: int, number of channels for the input feature
        """
        super(FCOSHead, self).__init__()
        self.num_classes = num_classes - 1

        cls_tower = []
        bbox_tower = []

        for i in range(cfg.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module("cls_tower", nn.Sequential(*cls_tower))
        self.add_module("bbox_tower", nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1, padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []

        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))

            bbox_tower = self.bbox_tower(feature)
            bbox_reg.append(torch.exp(self.scales[l](self.bbox_pred(bbox_tower))))

        return logits, bbox_reg, centerness