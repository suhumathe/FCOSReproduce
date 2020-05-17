# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:45:44 2020

@author: sks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Root directory of the project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly that they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

__C.TRAIN = edict()

# ---roidb setting related---
__C.TRAIN.USE_FLIPPED = True # whether use flipped images
__C.TRAIN.PROPOSAL_METHOD = 'gt' # Train with the boxes
__C.TRAIN.SCALES = (800,) # Pixel size of the shorter side of an image
__C.TRAIN.MAX_SIZE = 1333 # Max pixel size of the longest side of a scaled input image
# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True
# Trim size for input images to create minibatch
__C.TRAIN.TRIM_HEIGHT = 600
__C.TRAIN.TRIM_WIDTH = 600
# Maximal number of gt boxes in an image during training
__C.MAX_NUM_GT_BOXES = 20

# ---FCOS options---
__C.FCOS = edict()
__C.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128] # strides of features in different layers in the FPN
__C.FCOS.LOSS_ALPHA = 0.25 # Focal loss parameter: alpha
__C.FCOS.LOSS_GAMMA = 2.0 # Focal loss parameter: gamma

__C.FCOS.NUM_CONVS = 4 # number of conv blocks in FOCSHead

# ---resnet model options---
__C.RESNETS = edict()
__C.RESNETS.BACKBONE_OUT_CHANNELS = 256
__C.RESNETS.STEM_OUT_CHANNELS = 64 # output channels of base stem
__C.RESNETs.FIXED_BLOCKS = 2
__C.RESNETS.RES2_OUT_CHANNELS = 256

#---post process options---
__C.post.INFERENCE_TH = 0.05
__C.post.PRE_NMS_TOP_N = 1000
__C.post.NMS_TH = 0.6
__C.post.DETECTIONS_PER_IMG = 100

#---solver options: optimizer---
__C.SOLVER = edict()
__C.SOLVER.BASE_LR = 0.001
__C.SOLVER.WEIGHT_DECAY = 0.0005
__C.SOLVER.BIAS_LR_FACTOR = 2
__C.SOLVER.WEIGHT_DECAY_BIAS = 0
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.STEPS = (30000,)
__C.SOLVER.GAMMA = 0.1
__C.SOLVER.WARMUP_FACTOR = 1.0 / 3
__C.SOLVER.WARMUP_ITERS = 500
__C.SOLVER.WARMUP_METHOD = "linear"

import pdb
def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def get_output_tb_dir(imdb, weights_filename):
    """Return the directory where tensorflow summaries are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
