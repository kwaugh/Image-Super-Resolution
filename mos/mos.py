import os
import pickle
import random
import re
import subprocess
import sys
import tkinter as tk

from collections import defaultdict
from PIL import ImageTk, Image

seed = 1337
state_filename = 'mos_scores.pkl'
image_dirs = [
    'srgan_no_segs',
    'srgan_human_segs',
    'srgan_machine_segs',
    'srresnet_no_segs',
    'srresnet_human_segs',
    'srresnet_machine_segs'
]
image_types = ['bicubic', 'gen', 'lr', 'hr']
image_size = (384,384)
image_type_re = re.compile('({})\.png'.format('|'.join(image_types)))
gen_type_re = re.compile('({})'.format('|'.join(image_dirs)))
score_re = re.compile('^(([1-4](\.[0-9]+)?)|5)$')

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
    state = {
        'curr_idx': 0
    }
    for d in image_dirs:
        state['{}_average'.format(d)] = 0
        state['{}_count'.format(d)] = 0
    for t in image_types:
        if t != 'gen':
            state['{}_average'.format(t)] = 0
            state['{}_count'.format(t)] = 0
    write_state(state)
    return state

def write_state(state):
    with open(state_filename, 'wb') as fout:
        pickle.dump(state, fout, protocol=2)

def load_current_img(img_list, state):
    if state['curr_idx'] >= len(img_list):
        print("You rated everything. I don't believe it.")
        sys.exit()

    img_path = img_list[state['curr_idx']]
    img = Image.open(img_path)

    image_type = image_type_re.search(img_path).group(1)
    if image_type == 'hr':
        img = img.resize(image_size, resample=Image.BICUBIC)
    elif image_type == 'lr':
        img = img.resize(image_size, resample=Image.NEAREST)

    return img

def receive_score(img_list, state, tk_img_panel, tk_entry):
    raw_score = tk_entry.get()
    if not score_re.match(raw_score):
        tk_entry.delete(0, tk.END)
        return
    score = float(raw_score)

    img_path = img_list[state['curr_idx']]
    image_type = image_type_re.search(img_path).group(1)
    if image_type == 'gen':
        gen_type = gen_type_re.search(img_path).group(1)
        if gen_type:
            stat = gen_type
        else:
            print('Unknown filename format: {}. What did you do?'.format(img_path))
            sys.exit()
    else:
        stat = image_type

    cnt_key = '{}_count'.format(stat)
    avg_key = '{}_average'.format(stat)
    state[cnt_key] += 1
    state[avg_key] += (score - state[avg_key]) / state[cnt_key]
    state['curr_idx'] += 1
    write_state(state)

    tk_img = ImageTk.PhotoImage(load_current_img(img_list, state))
    tk_img_panel.configure(image=tk_img)
    tk_img_panel.image = tk_img

    tk_entry.delete(0, tk.END)

def main():
    random.seed(seed)

    state = load_state()

    img_list = []
    for d in image_dirs:
        img_list += load_file_list(d, regx='\.png')
    random.shuffle(img_list)

    window = tk.Tk()

    tk_img = ImageTk.PhotoImage(load_current_img(img_list, state))
    img_panel = tk.Label(window, image=tk_img)
    img_panel.pack()

    tk.Label(window, text='Rate image quality on a scale of 1-5:').pack()

    score_input = tk.Entry(window)
    score_input.bind("<Return>", lambda _: receive_score(img_list, state, img_panel, score_input))
    score_input.pack()
    score_input.focus_set()

    window.mainloop()

if __name__ == '__main__':
    main()
