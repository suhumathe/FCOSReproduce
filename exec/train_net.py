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
from exec.optimizer import make_lr_scheduler, make_optimizer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train the FCOS network")
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='res101, res152, etc',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=24, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models',
                        default='output', type=str)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='number of worker to load data',
                        default=1, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default=True, action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        default=True, action='store_true')
    parser.add_argument('--lscale', dest='lscale',
                        help='whether use large scale',
                        default=True, action='store_true')

    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='chechsession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)

    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)

    args = parser.parse_args()
    return args

def is_pytorch_1_1_0_or_later():
    return [int(_) for _ in torch.__version__.split(".")[:3]] >= [1, 1, 0]

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.dataset == 'pascal_voc':
        args.imdb_name = 'voc_2007_trainval'
        args.imdbval_name = 'voc_2007_test'
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '20']
    elif args.dataset == 'pascal_voc_0712':
        args.imdb_name = 'voc_2007_trainval + voc_2012_trainval'
        args.imdbval_name = 'voc_2007_test'
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '20']
    elif args.dataset == 'coco':
        args.imdb_name = 'coco_2014_train + coco_2014_valminusminival'
        args.imdbval_name = 'coco_2014_minival'
        args.set_cfgs = ['MAX_NUM_GT_BOXES', '20']

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_batch,
                                             num_workers=args.num_workers)

    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    det_net = FCOS(imdb.classes)
    optimizer = make_optimizer(det_net)
    scheduler = make_lr_scheduler(optimizer)
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()

    if args.resume:
        load_name = os.path.join(output_dir, 'fcos_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        det_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        det_net = nn.DataParallel(det_net)

    if args.cuda:
        det_net = det_net.cuda()

    iters_per_epoch = int(train_size / cfg.SOLVER.IMS_PER_BATCH)
    for epoch in range(args.start_epoch, args.max_epochs):
        det_net.train()
        loss_temp = 0
        start = time.time()

        if not pytorch_1_1_0_or_later:
            scheduler.step()

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = data_iter.next()
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
            det_net.zero_grad()

            loss_box_cls,  loss_box_reg, loss_centerness = det_net(im_data, im_info, gt_boxes)

            loss = loss_box_cls.mean() + loss_box_reg.mean() + loss_centerness.mean()
            loss_temp += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_box_cls = loss_box_cls.mean().item()
                    loss_box_reg = loss_box_reg.mean().item()
                    loss_centerness = loss_centerness.mean().item()
                else:
                    loss_box_cls = loss_box_cls.item()
                    loss_box_reg = loss_box_reg.item()
                    loss_centerness = loss_centerness.item()
                print("[session %d][epoch %2d][iter %4d] loss: %.4f, lr: %.2e" \
                      % (args.session, epoch, step, loss_temp, lr))
                print("\t\t\t loss_box_cls: %.4f, loss_box_reg: %.4f, loss_centerness: %.4f" \
                      % (loss_box_cls, loss_box_reg, loss_centerness))

                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_box_cls': loss_box_cls,
                        'loss_box_reg': loss_box_reg,
                        'loss_centerness': loss_centerness,
                    }

                loss_temp = 0
                start = time.time()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        if args.mGPUs:
            save_name = os.path.join(output_dir, 'fcos_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': det_net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_name)
        else:
            save_name = os.path.join(output_dir, 'fcos_{}_{}_{}.pth'.format(args.session, epoch, step))
            save_checkpoint({
                'session': args.session,
                'epoch': epoch + 1,
                'model': det_net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_name)
        print('save model: {}'.format(save_name))

        end = time.time()
        print(end - start)
