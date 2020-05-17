# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:53:13 2020

@author: sks
"""

import torch
import torch.nn as nn

INF = 100000000

class FCOSTarget(nn.Module):
    """
    Compute head targets, i.e., class label, regression, and centerness
    """
    def __init__(self):
        super(FCOSTarget, self).__init__()

    def compute_targets_for_locations(self, locations, gt_boxes, object_sizes_of_interest):
        """
        locations: [num_points, 2], center of the receptive filed of the feature points, obtained by FCOSLocation
        gt_boxes: [batch_size, num_boxes, 5], gt boxes provided in the image
        object_sizes_of_interest: [5, 2], ranges of object scales corresponding to each feature level
        returns class labels and regression targets for the feature points
        """
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        batch_size = gt_boxes.size(0)

        for i in range(batch_size):
            gt_boxes_per_im = gt_boxes[i, :, :4] # gt boxes for one image
            labels_per_im = gt_boxes[i, :, 4] # gt labels for one image
            area = (gt_boxes_per_im[:, 2] - gt_boxes_per_im[:, 0] + 1.) * (gt_boxes_per_im[:, 3] - gt_boxes_per_im[:, 1] + 1.)  # areas of gt boxes for one image
            area = area.type_as(gt_boxes)

            l = xs[:, None] - gt_boxes_per_im[:, 0][None, :] # [num_points, num_boxes]
            t = ys[:, None] - gt_boxes_per_im[:, 1][None, :] # [num_points, num_boxes]
            r = gt_boxes_per_im[:, 2][None, :] - xs[:, None] # [num_points, num_boxes]
            b = gt_boxes_per_im[:, 3][None, :] - ys[:, None] # [num_points, num_boxes]
            reg_targets_per_im = torch.stack([l, t, r ,b], dim=2).type_as(gt_boxes) # [num_points, num_boxes, 4]
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0 # [num_points, num_boxes] bool elements indicating whether the point in the gt box

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limite the regression range for each location
            is_cared_in_the_level = (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & (max_reg_targets_per_im >= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1) #[num_points, num_boxes]
            locations_to_gt_area[is_in_boxes==0] = INF
            locations_to_gt_area[is_cared_in_the_level==0] = INF

            # if there are still more than one objects for a location
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1) # num_points
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds] # [num_points, 4]
            labels_per_im = labels_per_im[locations_to_gt_inds] # [num_points, ]
            labels_per_im[locations_to_min_area==INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
        return labels, reg_targets

    def prepare_targets(self, points, gt_boxes):
        """
        calculate target labels and regression targets, and then transformed to facilitate the use of torch
        """
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = points.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)) # [num_points_per_level, 2]

        # points aggregation on all five feature maps
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0) # [num_points, 2]
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = self.compute_targets_for_locations(points, gt_boxes, expanded_object_sizes_of_interest)

        # points separation according to the number of points on each feature map
        batch_size = len(labels)
        for i in range(batch_size):
            labels_i = labels[i]
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        # points aggregation on the same level of images in the same batch
        labels_level_first = []
        reg_targets_level_first = []
        levels = len(points)
        for level in range(levels):
            labels_level_first.append(torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0))
            reg_targets_level_first.append(torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets]), dim=0)

        return labels_level_first, reg_targets_level_first

    def comput_centerness_targets(self, reg_targets):
        """
        compute centerness target for each points on the feature maps
        reg_targets: [num_points, 4] regression targets for each points
        """
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return centerness

    def forward(self, locations, gt_boxes):
        labels, reg_targets = self.prepare_targets(locations, gt_boxes)

        labels_flatten = []
        reg_targets_flatten = []
        levels = len(labels)
        for l in range(levels):
            # for each level in the feature pyramid
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))

        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten =  torch.cat(reg_targets_flatten, dim=0)

        pos_ind = torch.nonzero(labels_flatten>0).squeeze(1)

        if pos_ind.numel() > 0:
            centerness_targets = self.comput_centerness_targets(reg_targets_flatten)
        else:
            centerness_targets = None

        return labels_flatten, reg_targets_flatten, centerness_targets