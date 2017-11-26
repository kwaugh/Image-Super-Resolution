# Label definitions taken from:
# https://github.com/mcordts/cityscapesScripts

import numpy as np
import os

NUM_FEATURE_MAPS = 34

# takes in a greyscale label image of size (w x h)
# produces a (w x h x NUM_FEATURE_MAPS) image of one-hot label feature maps
# labels should be zero-indexed
def label_to_one_hot(im):
    im = np.clip(im, 0, NUM_FEATURE_MAPS)
    return (np.arange(NUM_FEATURE_MAPS) == im[...,None]).astype(np.int8)

def save_one_hot(data, path, name):
    np.save(os.path.join(path, name), data)

def load_one_hot(name, path=''):
    return np.load(os.path.join(path, name))
