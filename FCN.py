# Most of the code in this file was borrowed from https://github.com/shekkizh/FCN.tensorflow

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import scipy.misc as misc
import datetime
import auto_segment_helper as segment_helper
import TensorflowUtils as tf_utils
# import matplotlib.pyplot as plt

from six.moves import xrange
from utils import downsample_preserve_aspect_ratio_fn, get_imgs_fn, get_frame_key
from config import config
from tensorlayer.prepro import imresize

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "4", "batch size")
tf.flags.DEFINE_string("logs_dir", "segmentation_checkpoint/", "path to checkpoint directory")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_string("mode", "train", "Mode train/ valid")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

NUM_OF_CLASSESS = 151
IMAGE_SIZE = 256

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = tf_utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = tf_utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = tf_utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = tf_utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = tf_utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = tf_utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = tf_utils.max_pool_2x2(conv_final_layer)

        W6 = tf_utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = tf_utils.bias_variable([4096], name="b6")
        conv6 = tf_utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = tf_utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = tf_utils.bias_variable([4096], name="b7")
        conv7 = tf_utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = tf_utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = tf_utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = tf_utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = tf_utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = tf_utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = tf_utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = tf_utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = tf_utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = tf_utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = tf_utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = tf_utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = tf_utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    pred_annotation, _ = inference(image, keep_probability)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("Couldn't restore model.")

    if FLAGS.mode == "train":
        print("train")
        cfg = config.TRAIN
    elif FLAGS.mode == "valid":
        print("valid")
        cfg = config.VALID

    if not os.path.exists(cfg.segment_preprocessed_path):
        os.makedirs(cfg.segment_preprocessed_path)

    img_list = sorted(tl.files.load_file_list(path=cfg.hr_img_path, regx='.*.png', printable=False))
    rem = len(img_list) % FLAGS.batch_size
    for idx in range(0, len(img_list) - rem, FLAGS.batch_size):
        b_imgs_list = img_list[idx : idx + FLAGS.batch_size]
        load_image = lambda fn, path: downsample_preserve_aspect_ratio_fn(
                get_imgs_fn(fn, path), size=[256, 256])
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=load_image, path=cfg.hr_img_path)
        pred = sess.run(pred_annotation,
                feed_dict = {image: b_imgs, keep_probability: 1.0})

        # for i in range(FLAGS.batch_size):
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(b_imgs[0])
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(np.squeeze(pred[0].astype(np.uint8), axis=2))
        #     plt.savefig('compare_{}.png'.format(i))
        #     plt.show()
        # return

        for i in range(FLAGS.batch_size):
            img = pred[i].astype(np.uint8)
            img = imresize(img, size=[96, 96], interp='nearest', mode=None)
            img = np.squeeze(img, axis=2)
            one_hot = segment_helper.label_to_one_hot(img)
            segment_helper.save_one_hot(one_hot, cfg.segment_preprocessed_path,
                    get_frame_key(b_imgs_list[i]) + '_seg.npy')

        print('saved %d from %s' % (FLAGS.batch_size + idx, cfg.hr_img_path))

if __name__ == "__main__":
    assert(config.AUTO_SEGMENTATIONS)
    tf.app.run()
