# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:09:11 2020

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
import cv2
import cPickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from databuild.roidb import combined_roidb
from databuild.roibatchLoader import roibatchLoader
from cfgs.config import cfg, cfg_from_list, cfg_from_file, get_output_dir
from functional.net_utils import vis_detections

from nets.FCOS import FCOS
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def parse_args():
    parser = argparse.ArgumentParser(description='Test a FCOS network')
    parser.add_argument('--dataset', dest='dataset',
                        help='test dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys',
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default='./output', nargs=argparse.REMAINDER)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default=True, action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        default=False, action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        default=True, action='store_true')

    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load network',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=19, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=2504, type=int)

    parser.add_argument('--vis', dest='vis',
                        help='visulization mode',
                        default=False, action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    print('called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval + voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train + coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'fcos_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'fcos_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    det_net = FCOS(imdb.num_classes)

    print('load checkpoint %s' % (load_name))
    checkpoint = torch.load(load_name)
    det_net.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        det_net.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis
    save_name = 'fcos_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, cfg.TEST.IMS_PER_BATCH, \
                             imdb.num_classes, training=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)
    data_iter = iter(dataloader)
    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    det_net.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    for i in range(num_images):
        data = next(data_iter)
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        detections = det_net(im_data, im_info, gt_boxes)

        boxes = detections[:, :4]
        cls_labels = detections[:, 4]
        scores = detections[:, 5]

        boxes /= data[1][0][2].item()
        det_toc =  time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)

        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(cls_labels == j).contiguous().view(-1)

            if inds.numel() > 0:
                cls_scores = scores[inds]
                cls_boxes = boxes[inds, :]

                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                all_boxes[j][i] = cls_dets.cpu().numpy()

            else:
                all_boxes[j][i] = empty_array

        misc_toc = time.time()

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('images/result%d.png' % (i), im2show)
            pdb.set_trace()

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))








    