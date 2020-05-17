# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:25:35 2020

@author: sks
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from cfgs.config import cfg
from nets.backbone import build_backbone

from Layers.FCOSHead import FCOSHead
from Layers.FCOSLocation import FCOSLocation
from Layers.FCOSTarget import FCOSTarget
from Layers.FCOSLossComputation import FCOSLossComputation
from Layers.FCOSPostProcess import FCOSPostProcess

class FCOS(nn.Module):
    def __init__(self, num_classes):
        self.model = build_backbone()
        self.head = FCOSHead(num_classes, self.model.out_channels)
        self.target = FCOSTarget()
        self.loss = FCOSLossComputation()
        self.postProcess = FCOSPostProcess()
        self.locations = FCOSLocation()

    def forward(self, images, im_info, gt_boxes=None):
        if self.training and gt_boxes is None:
            raise ValueError("In training mode, targets should be passed")

        features = self.model(images)
        feature_shapes = []
        for feature in features:
            feature_shapes.append(feature.size())

        points_loc = self.locations(feature_shapes).type_as(gt_boxes)
        logits, bbox_reg, centerness = self.head(features)

        if self.training:
            return self._forward_train(points_loc, gt_boxes, logits, bbox_reg, centerness)
        else:
            return self._forward_test(logits, bbox_reg, centerness, im_info)

    def _forward_train(self, locations, gt_boxes, logits, bbox_reg, centerness):
        lables, reg_targets, centerness_targets = self.target(locations, gt_boxes)

        loss_box_cls, loss_box_reg, loss_centerness = self.loss(logits, bbox_reg, centerness, lables, reg_targets, centerness_targets)
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return losses

    def _forward_test(self, logits, bbox_reg, centerness, im_info):
        boxes = self.postProcess(logits, bbox_reg, centerness, im_info)
        return boxes
