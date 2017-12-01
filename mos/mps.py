import operator
import os
import pickle
import random
import re
import subprocess
import sys
import tkinter as tk

from collections import defaultdict, namedtuple
from PIL import ImageTk, Image

seed = 1337
state_filename = 'mps_scores.pkl'
image_dir_pairs = [
    ('srgan_no_segs', 'srgan_human_segs'),
    ('srresnet_no_segs', 'srresnet_human_segs'),
]
gen_type_re = re.compile('({})'.format('|'.join(map(operator.itemgetter(1), image_dir_pairs))))

def load_file_list(path, regx='\.png'):
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(os.path.join(path, f))
    return return_list

def load_state():
    if os.path.exists(state_filename):
        with open(state_filename, 'rb') as fin:
            return pickle.load(fin)
    state = { 'curr_idx': 0 }
    for p in image_dir_pairs:
        state['{}_average'.format(p[1])] = 0
        state['{}_count'.format(p[1])] = 0
    write_state(state)
    return state

def write_state(state):
    with open(state_filename, 'wb') as fout:
        pickle.dump(state, fout, protocol=2)

def load_current_img_pair(img_pair_list, state):
    if state['curr_idx'] >= len(img_pair_list):
        print("You rated everything. I don't believe it.")
        sys.exit()

    img_path_pair = img_pair_list[state['curr_idx']]
    return map(Image.open, img_path_pair)

def receive_key(sel, side, img_pair_list, state, left_img_panel, right_img_panel):
    img_path = img_pair_list[state['curr_idx']][1]
    gen_type = gen_type_re.search(img_path).group(1)
    if gen_type:
        stat = gen_type
    else:
        print('Unknown filename format: {}. What did you do?'.format(img_path))
        sys.exit()

    cnt_key = '{}_count'.format(stat)
    avg_key = '{}_average'.format(stat)
    state[cnt_key] += 1
    state[avg_key] += (int(sel == side) - state[avg_key]) / state[cnt_key]
    state['curr_idx'] += 1
    write_state(state)

    left_img_tk, right_img_tk = map(
            ImageTk.PhotoImage,
            load_current_img_pair(img_pair_list, state))
    left_img_panel.configure(image=left_img_tk)
    left_img_panel.image = left_img_tk
    right_img_panel.configure(image=right_img_tk)
    right_img_panel.image = right_img_tk

def main():
    random.seed(seed)

    state = load_state()

    img_pair_list = []
    for p in image_dir_pairs:
        list1 = load_file_list(p[0], regx='gen\.png')
        random.shuffle(list1)
        list2 = load_file_list(p[1], regx='gen\.png')
        random.shuffle(list2)
        img_pair_list += zip(list1, list2)
    random.shuffle(img_pair_list)

    window = tk.Tk()

    img_frame = tk.Frame()
    img_tk_pair = tuple(map(
            ImageTk.PhotoImage, 
            load_current_img_pair(img_pair_list, state)))
    random_side = random.randint(0, 1)
    left_img_panel = tk.Label(img_frame, image=img_tk_pair[int(not random_side)])
    right_img_panel = tk.Label(img_frame, image=img_tk_pair[random_side])
    left_img_panel.pack(side=tk.LEFT)
    right_img_panel.pack(side=tk.LEFT)
    img_frame.pack()

    instructions = 'Press left arrow key if the left image is of higher quality, or right arrow key if the right image is of higher quality.'
    tk.Label(window, text=instructions).pack()

    window.bind('<Left>', lambda _: receive_key(0, random_side, img_pair_list, state, left_img_panel, right_img_panel))
    window.bind('<Right>', lambda _: receive_key(1, random_side, img_pair_list, state, left_img_panel, right_img_panel))

    window.mainloop()

if __name__ == '__main__':
    main()
