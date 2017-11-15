import tensorlayer as tl
import numpy as np
import pickle
import segment_helper
from config import *
from utils import *
from PIL import Image

def save_to_disk(img, path, name):
    with open(os.path.join(path, name) + '.pickle', 'wb') as f_out:
        pickle.dump(img, f_out)

def load_from_disk(path, name):
    with open(os.path.join(path, name) + '.pickle', 'rb') as f_in:
        return pickle.load(f_in)

def save_all_segs(img_list, path='', save_path='', segment_suffix='.png', n_threads=4):
    segs = []
    segs_list = load_seg_file_list(img_list, config.TRAIN.segment_suffix)

    def save_seg_features(file_name, path, save_path):
        label_im = Image.open(os.path.join(path, file_name)).convert('RGB')
        w, h = label_im.size
        label_im = label_im.resize((w // 4, h // 4), resample=Image.NEAREST)
        save_to_disk(
                segment_helper.label_to_one_hot(label_im),
                save_path,
                file_name)

    rem = len(segs_list) % config.TRAIN.batch_size
    for idx in range(0, len(segs_list) - rem, n_threads):
        b_segs_list = segs_list[idx : idx + n_threads]
        tl.prepro.threading_data(b_segs_list, fn=save_seg_features, path=path, save_path=save_path)
        print('saved %d from %s' % (n_threads + idx, path))

if __name__ == '__main__':
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # save_all_segs(
    #         train_hr_img_list,
    #         path=config.TRAIN.segment_path,
    #         save_path=config.TRAIN.segment_preprocessed_path,
    #         segment_suffix=config.TRAIN.segment_suffix,
    #         n_threads=4)
    save_all_segs(
            valid_hr_img_list,
            path=config.VALID.segment_path,
            save_path=config.VALID.segment_preprocessed_path,
            segment_suffix=config.VALID.segment_suffix,
            n_threads=4)
