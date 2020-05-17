# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:15:08 2020

@author: sks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import Sampler


from databuild.roidb import combined_roidb
from databuild.roibatchLoader import roibatchLoader, sampler
from cfgs.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from functional.net_utils import weights_normal_init, save_net, load_net, adjust_learning_rate, save_checkpoint, clip_gradient
from nets.FCOS import FCOS

def parse_arg():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train the FCOS network")
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
