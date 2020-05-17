# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:20:54 2020

@author: sks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco

__sets = {}

for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())