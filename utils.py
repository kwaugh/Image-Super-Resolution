# Copyright (c) 2017 Hao Dong.
# Most code in this file was borrowed from https://github.com/zsdonghao/SRGAN

import os
import sys
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
from config import *

def get_imgs_fn(file_name, path, interp='bicubic'):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)

    return imresize(
            scipy.misc.imread(os.path.join(path, file_name), mode='RGB'),
            0.5,
            interp=interp,
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

def crop_square(x):
    h = len(x)
    w = len(x[0])
    min_dim = min(h, w)
    return crop(x, wrg=min_dim, hrg=min_dim, is_random=False)

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_preserve_aspect_ratio_fn(x):
    # Downsample to the right resolution, but also preserve the aspect ratio
    return downsample_fn(crop_square(x))

def upsample_fn(x):
    return imresize(x, size=[384, 384], interp='bicubic', mode=None)

# Given a cityscapes image path, remove the type and ext
# Input: dataset_dir/train/aachen_aachen_000000_000019_leftImg8bit.png
# Output: aachen_aachen_000000_000019
def get_frame_key(path):
    filename = os.path.basename(path)
    elems = filename.split('_')
    if len(elems) < 5:
        sys.stderr.write('Unknown image filename format: {}\n'.format(path))
    return '_'.join(elems[:4])

def load_seg_file_list(img_list, segment_suffix):
    files = []
    for img_path in img_list:
        prefix = get_frame_key(img_path)
        prefix = prefix[0:-16]
        seg_filename = '{}_{}'.format(prefix, segment_suffix)
        files.append(seg_filename)
    return files

# save all hr imgs to disk downsampled a bit
# save train and validation images
def save_hr_imgs():
    train_hr_img_list = sorted(tl.files.load_file_list(
        path=config.TRAIN.hr_img_path,
        regx='.*.png',
        printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.hr_img_path,
        regx='.*.png',
        printable=False))

    if not os.path.exists(os.path.join(config.TRAIN.hr_img_path, 'preprocessed')):
        os.makedirs(os.path.join(config.TRAIN.hr_img_path, 'preprocessed'))
    if not os.path.exists(os.path.join(config.VALID.hr_img_path, 'preprocessed')):
        os.makedirs(os.path.join(config.VALID.hr_img_path, 'preprocessed'))

    for i in range(len(train_hr_img_list)):
        img = get_imgs_fn(train_hr_img_list[i], config.TRAIN.hr_img_path)
        prepro_path = os.path.join(config.TRAIN.hr_img_path, 'preprocessed')
        np.save(os.path.join(prepro_path, train_hr_img_list[i]) + '_hr', img)
    for i in range(len(valid_hr_img_list)):
        img = get_imgs_fn(valid_hr_img_list[i], config.VALID.hr_img_path)
        prepro_path = os.path.join(config.VALID.hr_img_path, 'preprocessed')
        np.save(os.path.join(prepro_path, valid_hr_img_list[i]) + '_hr', img)

# save all lr imgs to disk
# only need to save validation images
def save_lr_imgs():
    valid_lr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.lr_img_path,
        regx='.*.png',
        printable=False))

    if not os.path.exists(os.path.join(config.VALID.lr_img_path, 'preprocessed')):
        os.makedirs(os.path.join(config.VALID.lr_img_path, 'preprocessed'))

    for i in range(len(valid_lr_img_list)):
        img = downsample_preserve_aspect_ratio_fn(get_imgs_fn(valid_lr_img_list[i], config.VALID.lr_img_path))
        prepro_path = os.path.join(config.VALID.lr_img_path, 'preprocessed')
        np.save(os.path.join(prepro_path, valid_lr_img_list[i]) + '_lr', img)
