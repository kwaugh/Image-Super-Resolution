import argparse
import os
import pickle
import random
import re
import subprocess

from collections import defaultdict, namedtuple

seed = 1337
state_filename = 'mos_scores.txt'
img_path = 'mos_images/'
image_types = [
    'srgan_no_segs',
    'srgan_human_segs',
    'srgan_machine_segs',
    'srresnet_no_segs',
    'srresnet_human_segs',
    'srresnet_machine_segs'
]

def load_file_list(path, regx='\.png'):
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    return return_list

def load_state():
    if os.path.exists(state_filename):
        with open(state_filename, 'rb') as fin:
            return pickle.load(fin)
    state = { 'curr_idx': 0 }
    for t in image_types:
        state['{}_average'.format(t)] = 0
        state['{}_count'.format(t)] = 0
    write_state(state)
    return state

def write_state(state):
    with open(state_filename, 'wb') as fout:
        pickle.dump(state, fout, protocol=2)

def collect_scores(path, viewer_app='eog'):
    state = load_state()

    img_list = load_file_list(path, regx='\.png')
    random.shuffle(img_list)

    for img_fn in img_list[state['curr_idx']:]:
        img_path = os.path.join(path, img_fn)
        viewer = subprocess.Popen([viewer_app, img_path])
        score = float(input('Rate image quality on a scale of 1-5: '))
        viewer.terminate()
        viewer.kill()

        img_type_re = re.compile('({})\.png'.format('|'.join(image_types)))
        img_type = img_type_re.search(img_fn).group(1)
        if img_type:
            cnt_key = '{}_count'.format(img_type)
            avg_key = '{}_average'.format(img_type)
            state[cnt_key] += 1
            state[avg_key] += (score - state[avg_key]) / state[cnt_key]

        state['curr_idx'] += 1
        write_state(state)

    print("You're all done. Send Keivaun or Paul your mos_scores.txt. Thanks!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', default='eog')
    args = parser.parse_args()

    random.seed(seed)
    collect_scores(img_path, viewer_app=args.viewer)

if __name__ == '__main__':
    main()
