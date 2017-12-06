import operator
import os
import pickle
import re

from collections import defaultdict

image_dirs = [
    'srgan_no_segs',
    'srgan_human_segs',
    'srgan_machine_segs',
    'srresnet_no_segs',
    'srresnet_human_segs',
    'srresnet_machine_segs'
]
mps_dirs = [
    'srgan_human_segs',
    'srresnet_human_segs',
]
image_types = ['bicubic', 'lr', 'hr']

def load_file_list(path, regx='\.png'):
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(os.path.join(path, f))
    return return_list

def load_state(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)

def main():
    mos_files = load_file_list('.', regx='mos_scores_.+\.pkl')
    mos_states = list(map(load_state, mos_files))
    total_counts = defaultdict(int)
    final_averages = defaultdict(int)

    for state in mos_states:
        for x in image_dirs + image_types:
            total_counts[x] += state['{}_count'.format(x)]

    for state in mos_states:
        for x in image_dirs + image_types:
            average = state['{}_average'.format(x)]
            weight = state['{}_count'.format(x)] / total_counts[x]
            final_averages[x] += average * weight

    scores = [(x, final_averages[x]) for x in image_dirs + image_types]
    scores.sort(key=operator.itemgetter(1), reverse=True)
    for score in scores:
        print('{} MOS score: {:.2f}'.format(*score))

    mps_files = load_file_list('.', regx='mps_scores_.+\.pkl')
    mps_states = list(map(load_state, mps_files))
    total_counts = defaultdict(int)
    final_averages = defaultdict(int)

    for state in mps_states:
        for x in mps_dirs:
            total_counts[x] += state['{}_count'.format(x)]

    for state in mps_states:
        for x in mps_dirs:
            average = state['{}_average'.format(x)]
            weight = state['{}_count'.format(x)] / total_counts[x]
            final_averages[x] += average * weight
    
    for x in mps_dirs:
        print('{} MPS score: {:.2f}'.format(x, final_averages[x]))

if __name__ == '__main__':
    main()
