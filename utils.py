# Copyright (c) 2017 Hao Dong.
# Most code in this file was borrowed from https://github.com/zsdonghao/SRGAN

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)

    return imresize(
            scipy.misc.imread(path + file_name, mode='RGB'),
            0.5,
            interp='bicubic',
            mode=None)

    # img = scipy.misc.imread(path + file_name, mode='RGB')
    # # resize the smallest dimension to be 384
    # h = len(img)
    # w = len(img[0])
    # min_dim = min(h, w)
    # scale_factor = float(384) / float(min_dim)
    # return crop(scipy.misc.imresize(img, size=scale_factor, interp='bicubic'), 384, 384)

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def upsample_fn(x):
    return imresize(x, size=[384, 384], interp='bicubic', mode=None)
