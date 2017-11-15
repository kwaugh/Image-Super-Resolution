# Copyright (c) 2017 Hao Dong.
# Most code in this file was borrowed from https://github.com/zsdonghao/SRGAN

from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 10 # originally was 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
# 80 epochs takes about 12 hours
config.TRAIN.n_epoch = 80 # originally was 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
# config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
# config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

# Streetview dataset
config.TRAIN.hr_img_path = '/media/kwaugh/RAID/Documents/visual_recognition/final_project/StreetView/leftImg8bit/train/'
# config.TRAIN.hr_img_path = '/work/05150/pchoi/maverick/leftImg8bit/train/'
config.TRAIN.lr_img_path = '/media/kwaugh/RAID/Documents/visual_recognition/final_project/StreetView/leftImg8bit/train/'
# config.TRAIN.segment_path = '/work/05150/pchoi/maverick/gtFine/train'
config.TRAIN.segment_path = '/media/kwaugh/RAID/Documents/visual_recognition/final_project/StreetView/gtFine/train/'
config.TRAIN.segment_suffix = 'gtFine_color.png'
config.TRAIN.segment_preprocessed_path = '/media/kwaugh/RAID/Documents/visual_recognition/final_project/StreetView/gtFine/train/preprocessed/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = '/media/kwaugh/RAID/Documents/visual_recognition/final_project/StreetView/leftImg8bit/val/'
# config.VALID.hr_img_path = '/work/05150/pchoi/maverick/leftImg8bit/val/'
config.VALID.lr_img_path = '/media/kwaugh/RAID/Documents/visual_recognition/final_project/StreetView/leftImg8bit/val/'
# config.VALID.lr_img_path = '/work/05150/pchoi/maverick/leftImg8bit/val/'
config.VALID.segment_path = '/media/kwaugh/RAID/Documents/visual_recognition/final_project/StreetView/gtFine/val/'
config.VALID.segment_suffix = 'gtFine_color.png'
config.VALID.segment_preprocessed_path = '/media/kwaugh/RAID/Documents/visual_recognition/final_project/StreetView/gtFine/val/preprocessed/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
