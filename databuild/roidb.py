# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:27:05 2020

@author: sks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL
import pdb

import datasets.imdb
from cfgs.config import cfg
from datasets.factory import get_imdb

def prepare_roidb(imdb):
    """
    Enrich the imdb roidb by adding some derived quantities that are useful for training.
    This function computes the maximum overlap, taken over gt boxes, between each RoI and each gt box.
    The class with maximum overlap is also recorded.
    """
    roidb = imdb.roidb
    if not (imdb.name.startswith('coco')):
        sizes = [PIL.Image.open(imdb.image_path_at(i)).size for i in range(imdb.num_images)]

    for i in range(len(imdb.image_index)):
        roidb[i]['img_id'] = imdb.image_id_at(i)
        roidb[i]['image'] = imdb.image_path_at(i)
        if not (imdb.name.startswith('coco')):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps

        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.

    ratio_list=[]
    for i in range(len(roidb)):
        width=roidb[i]['width']
        height=roidb[i]['height']
        ratio=width / float(height)

        if ratio>ratio_large:
            roidb[i]['need_crop']=1
            ratio=ratio_large
        elif ratio<ratio_small:
            roidb[i]['need_crop']=1
            ratio=ratio_small
        else:
            roidb[i]['need_crop']=0

        ratio_list.append(ratio)

    ratio_list=np.array(ratio_list)
    ratio_index=np.argsort(ratio_list)

    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box
    print('before filtering, there are %d images...' % (len(roidb)))
    for i in range(len(roidb)-1, -1, -1):
        if len(roidb[i]['boxes'])==0:
            del roidb[i]
    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):
    """
    Combine multiple roidbs
    """
    def get_training_roidb(imdb):
        """ return a roidb for train"""
        if cfg.TRAIN.USE_FLIPPED:
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')

        print('Preparing training data...')

        prepare_roidb(imdb)
        # ratio_index = rank_roidb_ratio(imdb)
        print('done')

        return imdb.roidb

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)

    ratio_list, ratio_index = rank_roidb_ratio(roidb)

    return imdb, roidb, ratio_list, ratio_index