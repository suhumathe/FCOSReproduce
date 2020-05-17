# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:36:19 2020

@author: sks
"""
import torch
from torch import nn
from torch.autograd import Variable
from cfgs.config import cfg


def sigmoid_focal_loss(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0]
    alpha = alpha[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    p = p.clamp(min=0.0001, max=1.0)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)

    loss = -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)
    return loss.sum()


def IoULoss(pred, target, weight=None):
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_interect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

    area_intersect = w_intersect * h_interect
    area_union = target_area + pred_area - area_intersect

    losses = -torch.log((area_intersect + 1.) / (area_union + 1.))

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum() / weight.sum()
    else:
        assert losses.numel() != 0
        return losses.mean()


class FCOSLossComputation(nn.Module):

    def __init__(self):
        super(FCOSLossComputation, self).__init__()
        self.FocalLoss_GAMMA = cfg.FCOS.LOSS_GAMMA
        self.FocalLoss_ALPHA = cfg.FCOS.LOSS_ALPHA

        self.cls_loss_func = sigmoid_focal_loss
        self.box_reg_loss_func = IoULoss
        self.centerness_loss_func = nn.BCEWithLogitsLoss()

    def forward(self, box_cls, box_regression, centerness, labels_flatten, reg_targets_flatten, centerness_targets):
        """
        box_cls, labels_faltten,: prediction and targets for box classification
        box_regression, reg_targets_flatten: prediction and targets for box regression
        centerness, centerness_targets: predictions and targets for point centerness
        return the three losses by using predefined loss functions
        """
        N = box_cls[0].size(0)  # batch_size
        num_classes = box_cls[0].size(1)  # classes
        levels = len(box_cls)  # levels

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []

        for l in range(levels):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        labels_flatten = Variable(labels_flatten.int())
        reg_targets_flatten = Variable(reg_targets_flatten)
        centerness_targets = Variable(centerness_targets)

        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten, self.FocalLoss_GAMMA, self.FocalLoss_ALPHA) / (
                    pos_inds.numel() + N)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            reg_loss = self.box_reg_loss_func(box_regression_flatten, reg_targets_flatten, centerness_targets)
            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets)
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss
