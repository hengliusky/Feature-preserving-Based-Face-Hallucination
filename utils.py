# coding:utf-8
import os
import random
import sys

import numpy as np
import scipy.misc
import tensorflow as tf
import tensorlayer as tl


def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    return scipy.misc.imread(path + file_name, mode='RGB')


def crop_sub_imgs_fn(x, is_random_crop=True):
    h, w = x.shape[:2]
    crop_h = crop_w = 148
    if is_random_crop:
        cx1 = random.randint(0, w - crop_w)
        cx2 = w - crop_w - cx1
        cy1 = random.randint(0, h - crop_h)
        cy2 = h - crop_h - cy1
    else:
        cx1 = cx2 = int(round((w - crop_w) / 2.))
        cy1 = cy2 = int(round((h - crop_h) / 2.))
    t = scipy.misc.imresize(
        x[cx1:h - cy1, cx2:w - cy2],
        size=[128, 128],
        interp='bicubic',
        mode=None)
    t = t / (255. / 2.)
    t = t - 1.
    return t


def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = scipy.misc.imresize(x, size=[16, 16], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def Res_block(n,
              n_filter=64,
              filter_size=(3, 3),
              strides=(1, 1),
              act=tf.nn.relu,
              padding='SAME',
              name='res_block'):
    nn = tl.layers.Conv2d(
        n,
        n_filter=n_filter,
        filter_size=filter_size,
        strides=strides,
        act=act,
        padding=padding,
        name=name + '/conv1')
    nn = tl.layers.Conv2d(
        nn,
        n_filter=n_filter,
        filter_size=filter_size,
        strides=strides,
        act=None,
        padding=padding,
        name=name + '/conv2')
    nn = tl.layers.ElementwiseLayer(
        [n, nn], combine_fn=tf.add, name=name + '/b_residual_add')
    nn.outputs = act(nn.outputs)
    return nn


def Pixelshuffler(network, scale=2, act=tf.nn.relu, name=''):
    network = tl.layers.Conv2d(
        network,
        n_filter=256,
        filter_size=3,
        strides=1,
        act=None,
        padding='SAME',
        name=name + '/conv')
    network = tl.layers.SubpixelConv2d(
        network,
        scale=scale,
        n_out_channel=None,
        act=act,
        name=name + '/pixelshuffler')
    return network
