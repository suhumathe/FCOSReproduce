# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:25:23 2020

@author: sks
"""
import torch
import numpy as np

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes

def clip_to_image(boxes, im_shape):
    TO_REMOVE = 1
    boxes[:, 0].clamp_(min=0, max=im_shape[0] - TO_REMOVE)
    boxes[:, 1].clamp_(min=0, max=im_shape[1] - TO_REMOVE)
    boxes[:, 2].clamp_(min=0, max=im_shape[0] - TO_REMOVE)
    boxes[:, 3].clamp_(min=0, max=im_shape[1] - TO_REMOVE)

    return boxes

def remove_small_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = ((ws >= min_size) & (hs >= min_size))

    return boxes[keep]

def bbox_overlaps(boxes, gt_boxes):
    """
    --Inputs--
    boxes: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    --Outputs--
    overlaps: (N, K) ndarray of float, overlap between boxes and gt_boxes
    """
    N = boxes.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1.) * (gt_boxes[:,3] - gt_boxes[:,1]+1.)).view(1,K)
    boxes_area = ((boxes[:,2] - boxes[:,0] + 1.) * (boxes[:,3] - boxes[:,1]+1.)).view(N,1)

    expand_boxes = boxes.view(N,1,4).expand(N,K,4)
    expand_gt_boxes = gt_boxes.view(1,K,4).expand(N,K,4)

    iw = (torch.min(expand_boxes[:,:,2], expand_gt_boxes[:,:,2]) - torch.max(expand_boxes[:,:,0], expand_gt_boxes[:,:,0]) + 1.)
    iw[iw < 0] = 0.

    ih = (torch.min(expand_boxes[:,:,3], expand_gt_boxes[:,:,3]) - torch.max(expand_boxes[:,:,1], expand_gt_boxes[:,:,1]) + 1.)
    ih[ih < 0] = 0.

    ua = gt_boxes_area + boxes_area - iw * ih
    overlaaps = iw * ih / ua

    return overlaaps

def bbox_overlaps_batch(boxes, gt_boxes):
    """
    --Inputs--
    boxes : (N, 4) or (b, N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    --Outputs--
    overlaps: (b, N, K) ndarray of flaot,  overlap between boxes and gt_boxes
    """
    batch_size = gt_boxes.size(0)

    if boxes.dim() == 2:

        N = boxes.size(0)
        K = gt_boxes.size(1)

        boxes = boxes.view(1,N,4).expand(batch_size,N,4).contiguous() # (b,N,4)
        gt_boxes = gt_boxes[:,:,:4].contiguous() # (b,K,4)

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1.) # (b,K)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1.) # (b,K)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size,1,K) # (b,1,K)

        boxes_x = (boxes[:,:,2] - boxes[:,:,0] + 1.) # (b,N)
        boxes_y = (boxes[:,:,3] - boxes[:,:,1] + 1.) # (b,N)
        boxes_area = (boxes_x * boxes_y).view(batch_size,N,1) # (b,N,1)

        gt_boxes_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) # (b,K)
        boxes_area_zero = (boxes_x == 1) & (boxes_y ==1) # (b,N)

        expand_boxes = boxes.view(batch_size,N,1,4).expand(batch_size,N,K,4) # (b,N,K,4)
        expand_gt_boxes = gt_boxes.view(batch_size,1,K,4).expand(batch_size,N,K,4) # (b,N,K,4)

        iw = (torch.min(expand_boxes[:,:,:,2], expand_gt_boxes[:,:,:,2]) - torch.max(expand_boxes[:,:,:,0], expand_gt_boxes[:,:,:,0]) + 1.) # (b,N,K)
        iw[iw < 0] = 0.

        ih = (torch.min(expand_boxes[:,:,:,3], expand_gt_boxes[:,:,:,3]) - torch.max(expand_boxes[:,:,:,0], expand_gt_boxes[:,:,:,0]) + 1.) # (b,N,K)
        ih[ih < 0] = 0.

        ua = gt_boxes_area + boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        overlaps.masked_fill_(gt_boxes_area_zero.view(batch_size,1,K).expand(batch_size,N,K), 0)
        overlaps.masked_fill_(boxes_area_zero.view(batch_size,N,1).expand(batch_size,N,K), -1)

    elif boxes.dim() == 3:
        N = boxes.size(1)
        K = gt_boxes.size(1)

        if boxes.size(2) == 4:
            boxes = boxes[:,:,:4].contiguous()
        else:
            boxes = boxes[:,:,:4].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1.) # (b,K)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1.) # (b,K)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size,1,K) # (b,1,K)

        boxes_x = (boxes[:,:,2] - boxes[:,:,0] + 1.) # (b,N)
        boxes_y = (boxes[:,:,3] - boxes[:,:,1] + 1.) # (b,N)
        boxes_area = (boxes_x * boxes_y).view(batch_size,N,1) # (b,N,1)

        gt_boxes_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) # (b,K)
        boxes_area_zero = (boxes_x == 1) & (boxes_y == 1) # (b,N)

        expand_boxes = boxes.view(batch_size,N,1,4).expand(batch_size,N,K,4)
        expand_gt_boxes = gt_boxes.view(batch_size,1,K,4).expand(batch_size,N,K,4)

        iw = (torch.min(expand_boxes[:,:,:,2], expand_gt_boxes[:,:,:,2]) - torch.max(expand_boxes[:,:,:,0], expand_gt_boxes[:,:,:,0]) + 1.) # (b,N,K)
        iw[iw < 0] = 0.

        ih = (torch.min(expand_boxes[:,:,:,3], expand_gt_boxes[:,:,:,3]) - torch.max(expand_boxes[:,:,:,1], expand_gt_boxes[:,:,:,1]) + 1.) # (b,N,K)
        ih[ih < 0] = 0.

        ua = gt_boxes_area + boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        overlaps.masked_fill_(gt_boxes_area_zero.view(batch_size,1,K).expand(batch_size,N,K), 0)
        overlaps.masked_fill_(boxes_area_zero.view(batch_size,N,1).expand(batch_size,N,K), -1)
    else:
        raise ValueError('input dimension is not correct.')

    return overlaps