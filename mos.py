import argparse
import os
import random
import re
import subprocess

from collections import defaultdict

evaluate_dirs = [
    'evaluate-srgan_False_False',
    'evaluate-srgan_True_False',
    'evaluate-srresnet_True_True',
]

image_types = ['bicubic', 'gen', 'hr', 'lr']

def load_file_list(path, regx='\.png'):
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    return return_list

def collect_scores(path, viewer_app='eog'):
    score_filename = '{}_scores.txt'.format(path)
    if os.path.exists(score_filename):
        return

    img_list = load_file_list(path, regx='\.png')
    random.shuffle(img_list)

    averages = defaultdict(int)
    counts = defaultdict(int)

    for img_fn in img_list:
        img_path = os.path.join(path, img_fn)
        viewer = subprocess.Popen([viewer_app, img_path])
        score = float(input('Rate image quality on a scale of 1-5: '))
        viewer.terminate()
        viewer.kill()

        img_type_re = re.compile('({})\.png'.format('|'.join(image_types)))
        img_type = img_type_re.search(img_fn).group(1)
        if img_type:
            counts[img_type] += 1
            averages[img_type] += (score - averages[img_type]) / counts[img_type]

    score_file = open(score_filename, 'w')
    for t in image_types:
        score_file.write('average {}: {}\n'.format(t, averages[t]))
    score_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', default='eog')
    args = parser.parse_args()

    for d in evaluate_dirs:
        collect_scores(d, viewer_app=args.viewer)

if __name__ == '__main__':
    main()
