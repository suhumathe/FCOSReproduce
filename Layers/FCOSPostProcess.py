# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:08:34 2020

@author: sks
"""
import torch
import torch.nn as nn

from cfgs.config import cfg
from functional.bbox import clip_to_image, remove_small_boxes
from functional.nms import nms

class FCOSPostProcess(nn.Module):
    """
    Perform post-processing on the outputs of the boxes.
    This is only used in inference
    """
    def __init__(self, min_size, num_classes):

        self.pre_nms_thresh = cfg.post.INFERENCE_TH
        self.pre_nms_top_n = cfg.post.PRE_NMS_TOP_N
        self.nms_thresh = cfg.post.NMS_TH
        self.post_nms_top_n = cfg.post.DETECTIONS_PER_IMG

        self.min_size = min_size
        self.num_classes = num_classes

    def forward_for_single_feature_map(self, locations, box_cls, box_regression, centerness, im_info):
        """
        the inputs variables are for one feature map, locations are the RF centers of the points,
        box_cls, box_regression and centerness are the outputs of FCOS head

        locations: list of tensor, [num_points, 2], RF centers of points
        box_cls: list of tensor, [batch_size, num_classes, h, w], classification results
        box_regression: list of tensor, [batch_size, 4, h, w], regression results
        centerness: list of tensor, [batch_size, 1, h, w], centerness results
        image_sizes: tuple, [2, ], height and width of the image
        return: detection boxes, their corresponding labels and confidence scores
        """
        N, C, H, W = box_cls.shape # batch_size, num_classes, h, and w

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1) # [batch_size, h, w, C]
        box_cls = box_cls.reshape(N, -1, C).sigmoid() # [batch_size, num_points, C] num_points is the number of points in locations
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1) # [batch_size, h, w, 4]
        box_regression = box_regression.reshape(N, -1, 4) # [batch_size, num_points, 4] num_points is the number of points in locations
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1) # [batch_size, h, w, 1]
        centerness = centerness.reshape(N, -1).sigmoid() # [batch_size, num_points]

        candidate_inds = box_cls > self.pre_nms_thresh # [batch_size, num_points, C]
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1) # [batch_size,]
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i] # [num_points, C]
            per_candidate_inds = candidate_inds[i] # [num_points, C], bool values indicate confidence scores whether exceed the threshold
            per_box_cls = per_box_cls[per_candidate_inds] # [n_scores, ], 包含不同类别的同一点

            per_candidate_nonzero = per_candidate_inds.nonzero() # [n_scores, 2]
            per_box_loc = per_candidate_nonzero[:, 0] # [n_scores, ], index of points
            per_class = per_candidate_nonzero[:, 1] + 1 # [n_scores, ], index of classes

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc] # [n_scores, 4], regression targets for selected locations
            per_locations = locations[per_box_loc] # [n_scores, 2], selected locations

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = per_box_loc.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = im_info[i, 0], im_info[i, 1]
            boxlist = torch.cat([detections, per_class[:, None], per_box_cls[:, None]])
            boxlist = clip_to_image(boxlist, (int(w), int(h)))
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        """
        boxlist: list of tensors, num_imgs*[num_boxes, 5]
        """
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            boxes = boxlists[i][:, :4]
            labels = boxlists[i][:, 4]
            scores = boxlists[i][:, 5]
            result = []

            # class specific detections
            for j in range(1, self.num_classes):
                inds = (labels==j).nonzero().view(-1)
                scores_j = scores[inds]
                boxes_j = boxes[inds, :].view(-1, 4)

                _, order = torch.sort(scores_j, 0, True)
                keep = nms(boxes_j[order, :], scores_j[order], self.nms_thresh)
                keep = keep.view(-1).long()
                scores_j = scores_j[keep]
                boxes_j = boxes_j[keep, :]
                num_labels = len(scores_j)
                labels_j = torch.full((num_labels,), j, dtype=torch.int64, device=scores.device)
                box_list_for_class = torch.cat([boxes_j, labels_j[:, None], scores_j[:, None]], dim=1)
                result.append(box_list_for_class)

            result = torch.cat(result, dim=0)
            number_of_detections = len(result)

            # limit to max_per_image detections, over all classes
            if number_of_detections > self.post_nms_top_n >0:
                cls_score = result[:, 5]
                image_thresh, _ =torch.kthvalue(cls_score.cpu(), number_of_detections-self.post_nms_top_n+1)
                keep = cls_score>=image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

    def forward(self, locations, box_cls, box_regression, centerness, im_info):
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(self.forward_for_single_feature_map(l, o, b, c, im_info))

        boxlists = []
        num_images = len(sampled_boxes[0])
        for n in range(num_images):
            box_lists_per_image = [sampled_boxes_per_level[n] for sampled_boxes_per_level in sampled_boxes]
            box_lists_per_image = torch.cat(box_lists_per_image, dim=0)
            boxlists.append(box_lists_per_image)

        boxlists = self.select_over_all_levels(boxlists)
        return boxlists