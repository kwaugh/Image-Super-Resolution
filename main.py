# Copyright (c) 2017 Hao Dong.
# Most code in this file was borrowed from https://github.com/zsdonghao/SRGAN

#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
import segment_helper
from model import *
from utils import *
from config import config, log_config
from PIL import Image

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def read_all_imgs(img_list, path='', n_threads=4):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    # remove extra so that we have full batches
    rem = len(img_list) % config.TRAIN.batch_size
    for idx in range(0, len(img_list) - rem, n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def read_all_imgs_bicubic(img_list, path='', n_threads=4):
    """ Returns all images in array by given path and name of each image file. """
    """ Downscales the image by 4x using bicubic interpolation"""
    imgs = []
    # remove extra so that we have full batches
    rem = len(img_list) % config.TRAIN.batch_size
    for idx in range(0, len(img_list) - rem, n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(
            b_imgs_list,
            fn=lambda file_name, path: downsample_preserve_aspect_ratio_fn(get_imgs_fn(file_name, path)),
            path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def read_all_segs(seg_list, path='', n_threads=4):
    """ Loads all preprocessed segs into memory. """
    segs = []
    rem = len(seg_list) % config.TRAIN.batch_size
    for idx in range(0, len(seg_list) - rem, n_threads):
        b_segs_list = seg_list[idx : idx + n_threads]
        b_segs = tl.prepro.threading_data(
            b_segs_list,
            fn=lambda file_name, path: segment_helper.load_one_hot(file_name, path),
            path=path)
        segs.extend(b_segs)
        print('read %d from %s' % (len(segs), path))
    return segs


def train_srgan():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_{}_ginit".format(tl.global_flag['mode'], tl.global_flag['use_segs'])
    save_dir_gan = "samples/{}_{}_gan".format(tl.global_flag['mode'], tl.global_flag['use_segs'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    if (tl.global_flag['use_segs']):
        train_segs_list = sorted(tl.files.load_file_list(
            path=config.TRAIN.segment_preprocessed_path,
            regx='.*.npy',
            printable=False))
    train_hr_img_list = sorted(tl.files.load_file_list(
        path=config.TRAIN.hr_img_path,
        regx='.*.png',
        printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))[:8]
    valid_hr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.hr_img_path,
        regx='.*.png',
        printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.lr_img_path,
        regx='.*.png',
        printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=4)
    if (tl.global_flag['use_segs']):
        train_segs_imgs = read_all_segs(train_segs_list, path=config.TRAIN.segment_preprocessed_path, n_threads=4)
    valid_lr_imgs = read_all_imgs_bicubic(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=4)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=4)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    if (tl.global_flag['use_segs']):
        t_seg = tf.placeholder(
                'float32',
                [batch_size, 96, 96, segment_helper.NUM_FEATURE_MAPS],
                name='t_seg_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')

    if (tl.global_flag['use_segs']):
        net_g = SRGAN_g_seg(t_image, t_seg, is_train=True, reuse=False)
    else:
        net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False)
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
    if (tl.global_flag['use_segs']):
        net_g_test = SRGAN_g_seg(t_image, t_seg, is_train=False, reuse=True)
    else:
        net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.global_flag['loaded_weights'] = False
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_{}.npz'.format(tl.global_flag['mode'], tl.global_flag['use_segs']), network=net_g) is False:
        if tl.files.load_and_assign_npz(
                sess=sess,
                name=checkpoint_dir+'/g_{}_{}_init.npz'.format(tl.global_flag['mode'], tl.global_flag['use_segs']),
                network=net_g) is not False:
            tl.global_flag['loaded_weights'] = True
    else:
        tl.global_flag['loaded_weights'] = True
    tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir+'/d_{}_{}.npz'.format(tl.global_flag['mode'], tl.global_flag['use_segs']),
            network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    if (tl.global_flag['use_segs']):
        sample_segs = train_segs_imgs[0:batch_size]
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=4) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:',sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    if (tl.global_flag['use_segs']):
        sample_segs_96 = tl.prepro.threading_data(sample_segs, fn=lambda x: x)
        print('sample segs sub-image:', sample_segs_96.shape, sample_segs_96.min(), sample_segs_96.max())
    # tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit+'/_train_sample_96.png')
    # tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit+'/_train_sample_384.png')
    # tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan+'/_train_sample_96.png')
    # tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan+'/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    # only init if we haven't loaded weights
    if not tl.global_flag['loaded_weights']:
        for epoch in range(0, n_epoch_init+1):
            epoch_time = time.time()
            total_mse_loss, n_iter = 0, 0

            ## If your machine cannot load all images into memory, you should use
            ## this one to load batch of images while training.
            # random.shuffle(train_hr_img_list)
            # for idx in range(0, len(train_hr_img_list), batch_size):
            #     step_time = time.time()
            #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
            #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            ## If your machine have enough memory, please pre-load the whole train set.
            for idx in range(0, len(train_hr_imgs), batch_size):
                step_time = time.time()
                b_imgs_384 = tl.prepro.threading_data(
                        train_hr_imgs[idx : idx + batch_size],
                        fn=crop_sub_imgs_fn, is_random=True)
                if (tl.global_flag['use_segs']):
                    b_segs = tl.prepro.threading_data(
                            train_segs_imgs[idx : idx + batch_size],
                            fn=lambda x: x)
                b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
                ## update G
                if (tl.global_flag['use_segs']):
                    errM, _ = sess.run(
                            [mse_loss, g_optim_init],
                            {t_image: b_imgs_96, t_target_image: b_imgs_384, t_seg: b_segs})
                else:
                    errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f "
                        % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                total_mse_loss += errM
                n_iter += 1
            print("[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f"
                    % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter))

            ## quick evaluation on train set
            if (epoch != 0) and (epoch % 10 == 0):
                if (tl.global_flag['use_segs']):
                    out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96, t_seg: sample_segs_96})
                else:
                    out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
                # print("[*] save images")
                # tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)

            ## save model
            if (epoch != 0) and (epoch % 10 == 0):
                tl.files.save_npz(
                        net_g.all_params,
                        name=checkpoint_dir+'/g_{}_{}_init.npz'.format(tl.global_flag['mode'], tl.global_flag['use_segs']),
                        sess=sess)

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        rem = len(train_hr_imgs) % config.TRAIN.batch_size
        for idx in range(0, len(train_hr_imgs) - rem, batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            if (tl.global_flag['use_segs']):
                b_segs = tl.prepro.threading_data(
                        train_segs_imgs[idx : idx + batch_size],
                        fn=lambda x: x)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update D
            if (tl.global_flag['use_segs']):
                errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384, t_seg: b_segs})
            else:
                errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            ## update G
            if (tl.global_flag['use_segs']):
                errG, errM, errV, errA, _ = sess.run(
                        [g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim],
                        {t_image: b_imgs_96, t_target_image: b_imgs_384, t_seg: b_segs})
            else:
                errG, errM, errV, errA, _ = sess.run(
                        [g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim],
                        {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)"
                    % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        print("[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f"
                % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter))

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            if (tl.global_flag['use_segs']):
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96, t_seg: sample_segs_96})
            else:
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
            # print("[*] save images")
            # tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 5 == 0):
            tl.files.save_npz(
                    net_g.all_params,
                    name=checkpoint_dir+'/g_{}_{}.npz'.format(tl.global_flag['mode'], tl.global_flag['use_segs']),
                    sess=sess)
            tl.files.save_npz(
                    net_d.all_params,
                    name=checkpoint_dir+'/d_{}_{}.npz'.format(tl.global_flag['mode'], tl.global_flag['use_segs']),
                    sess=sess)

def train_srresnet():
    ## create folders to save result images and trained model
    save_dir = "samples/{}_{}_resnet".format(tl.global_flag['mode'], tl.global_flag['use_segs'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    if (tl.global_flag['use_segs']):
        train_segs_list = sorted(tl.files.load_file_list(
            path=config.TRAIN.segment_preprocessed_path,
            regx='.*.npy',
            printable=False))
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=4)
    valid_lr_imgs = read_all_imgs_bicubic(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=4)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=4)
    if (tl.global_flag['use_segs']):
        train_segs_imgs = read_all_segs(train_segs_list, path=config.TRAIN.segment_preprocessed_path, n_threads=4)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_SRRESNET')
    if (tl.global_flag['use_segs']):
        t_seg = tf.placeholder(
                'float32',
                [batch_size, 96, 96, segment_helper.NUM_FEATURE_MAPS],
                name='t_seg_input_to_SRRESNET_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')

    if (tl.global_flag['use_segs']):
        net_g = SRGAN_g_seg(t_image, t_seg, is_train=True, reuse=False)
    else:
        net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_g.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False)
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
    if (tl.global_flag['use_segs']):
        net_g_test = SRGAN_g_seg(t_image, t_seg, is_train=False, reuse=True)
    else:
        net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    ###========================== DEFINE TRAIN OPS ==========================###
    mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    g_loss = mse_loss + vgg_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(
            sess=sess,
            name=checkpoint_dir+'/g_{}_{}.npz'.format(tl.global_flag['mode'], tl.global_flag['use_segs']),
            network=net_g)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    if (tl.global_flag['use_segs']):
        sample_segs = train_segs_imgs[0:batch_size]
    # if no pre-load train set
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=4)
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:',sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    if (tl.global_flag['use_segs']):
        sample_segs_96 = tl.prepro.threading_data(sample_segs, fn=lambda x: x)
        print('sample segs sub-image:', sample_segs_96.shape, sample_segs_96.min(), sample_segs_96.max())
    # tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit+'/_train_sample_96.png')
    # tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit+'/_train_sample_384.png')
    # tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan+'/_train_sample_96.png')
    # tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan+'/_train_sample_384.png')

    ###========================= train G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for ResNet)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            print(" ** init lr: %f  decay_every_init: %d, lr_decay: %f (for ResNet)"
                    % (lr_init, decay_every, lr_decay))

        epoch_time = time.time()
        total_g_loss, n_iter = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            if (tl.global_flag['use_segs']):
                b_segs = tl.prepro.threading_data(
                        train_segs_imgs[idx : idx + batch_size],
                        fn=lambda x: x)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update G
            if (tl.global_flag['use_segs']):
                errG, errM, errV, _ = sess.run(
                        [g_loss, mse_loss, vgg_loss, g_optim],
                        {t_image: b_imgs_96, t_target_image: b_imgs_384, t_seg: b_segs})
            else:
                errG, errM, errV, _ = sess.run(
                        [g_loss, mse_loss, vgg_loss, g_optim],
                        {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, g_loss: %.8f (mse: %.6f vgg: %.6f)"
                    % (epoch, n_epoch, n_iter, time.time() - step_time, errG, errM, errV))
            total_g_loss += errG
            n_iter += 1

        print("[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f"
                % (epoch, n_epoch, time.time() - epoch_time, total_g_loss/n_iter))

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            if (tl.global_flag['use_segs']):
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96, t_seg: sample_segs_96})
            else:
                out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
            # print("[*] save images")
            # tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(
                    net_g.all_params,
                    name=checkpoint_dir+'/g_{}_{}.npz'.format(tl.global_flag['mode'], tl.global_flag['use_segs']),
                    sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}_{}".format(tl.global_flag['mode'], tl.global_flag['use_segs'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    if tl.global_flag['use_segs']:
        valid_segs_list = sorted(tl.files.load_file_list(
                    path=config.VALID.segment_preprocessed_path,
                    regx='.*.npy',
                    printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.hr_img_path,
        regx='.*.png',
        printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.lr_img_path,
        regx='.*.png',
        printable=False))

    print('valid_hr_img_list:', len(valid_hr_img_list))
    print('valid_lr_img_list:', len(valid_lr_img_list))
    if tl.global_flag['use_segs']:
        print('valid_segs_list:', len(valid_segs_list))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.VALID.hr_img_path, n_threads=4)
    # for im in train_hr_imgs:
    #     print(im.shape)
    if tl.global_flag['use_segs']:
        valid_segs_imgs = read_all_segs(valid_segs_list, path=config.VALID.segment_preprocessed_path, n_threads=4)
    valid_lr_imgs = read_all_imgs_bicubic(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=4)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=4)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = len(valid_lr_imgs) // 2 # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1   # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
    # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    if tl.global_flag['use_segs']:
        t_seg = tf.placeholder(
                'float32',
                [None, 96, 96, segment_helper.NUM_FEATURE_MAPS],
                name='t_seg_input_to_SRGAN_generator')

    if tl.global_flag['use_segs']:
        net_g = SRGAN_g_seg(t_image, t_seg, is_train=False, reuse=False)
    else:
        net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    checkpoint_file = '/g_{}_{}.npz'.format(tl.global_flag['mode'].split('-')[1], tl.global_flag['use_segs'])
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+checkpoint_file, network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    mse_gen = 0 # mean squared error
    mse_bicubic = 0 # mean squared error
    h = valid_lr_img.shape[0]
    w = valid_lr_img.shape[1]
    for i in range(len(valid_lr_imgs)):
        valid_lr_img = valid_lr_imgs[i]
        valid_hr_img = crop_square(valid_hr_imgs[i])
        if tl.global_flag['use_segs']:
            valid_seg_img = valid_segs_imgs[i]
        if tl.global_flag['use_segs']:
            out = sess.run(net_g.outputs, {t_image: [valid_lr_img], t_seg: [valid_seg_img]})
        else:
            out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        print("took: %4.4fs" % (time.time() - start_time))

        # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("LR size: %s /  generated HR size: %s" % (size, out.shape))
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir+'/valid_'+str(i)+'gen.png')
        tl.vis.save_image(valid_lr_img, save_dir+'/valid_'+str(i)+'lr.png')
        tl.vis.save_image(valid_hr_img, save_dir+'/valid_'+str(i)+'hr.png')

        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, save_dir+'/valid_'+str(i)+'bicubic.png')

        resized_hr_img = scipy.misc.imresize(valid_hr_img, [384, 384], interp='bicubic', mode=None)
        # print((out[0] * 255).astype("int"))
        mse_gen += np.sum((resized_hr_img.astype("float") - out[0].astype("float")) ** 2)
        mse_bicubic += np.sum((resized_hr_img.astype("float") - out_bicu.astype("float")) ** 2)

    mse_gen /= w * h * len(valid_lr_img) * 3 # 3 because or rgb channels
    mse_bicubic /= w * h * len(valid_lr_img) * 3 # 3 because or rgb channels
    print("Mean_squared_error_gen: " + str(mse_gen))
    print("Mean_squared_error_bicubic: " + str(mse_bicubic))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan',
            help='srgan, srresnet, evaluate-srgan, evaluate-srresnet')

    parser.add_argument('--use-segs', type=str, default='True',
            help='Use segmentations or not')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['use_segs'] = args.use_segs == 'True'

    if tl.global_flag['mode'] == 'srgan':
        train_srgan()
    elif tl.global_flag['mode'] == 'srresnet':
        train_srresnet()
    elif (tl.global_flag['mode'] == 'evaluate-srgan' or
          tl.global_flag['mode'] == 'evaluate-srresnet'):
        evaluate()
    else:
        raise Exception("Unknown --mode: {}".format(args.mode))
