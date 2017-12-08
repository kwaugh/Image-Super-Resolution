# Copyright (c) 2017 Hao Dong.
# Most code in this file was borrowed from https://github.com/zsdonghao/SRGAN

from easydict import EasyDict as edict
import json

config = edict()
config.AUTO_SEGMENTATIONS = True
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 10 # originally was 100

## adversarial learning (SRGAN)
# 80 epochs takes about 12 hours
config.TRAIN.n_epoch = 80 # originally was 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

# Streetview dataset
config.TRAIN.hr_img_path = 'leftImg8bit/train/'
if config.AUTO_SEGMENTATIONS:
    config.TRAIN.segment_preprocessed_path = 'auto_segmentations/train/preprocessed'
else:
    config.TRAIN.segment_preprocessed_path = 'gtFine/train/preprocessed'
config.TRAIN.cityscapes_segment_path = 'gtFine/train'
config.TRAIN.cityscapes_segment_suffix = 'gtFine_color.png'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'leftImg8bit/val/'
config.VALID.lr_img_path = 'leftImg8bit/val/'
if config.AUTO_SEGMENTATIONS:
    config.VALID.segment_preprocessed_path = 'auto_segmentations/val/preprocessed/'
else:
    config.VALID.segment_preprocessed_path = 'gtFine/val/preprocessed/'
config.VALID.cityscapes_segment_path = 'gtFine/val/'
config.VALID.cityscapes_segment_suffix = 'gtFine_color.png'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
