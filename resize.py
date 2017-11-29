import os
import re

from collections import defaultdict
from PIL import Image

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

def collect_scores(path, size):
    img_list = load_file_list(path, regx='\.png')[:4]

    for img_fn in img_list:
        img_path = os.path.join(path, img_fn)
        img = Image.open(img_path)

        img_type_re = re.compile('({})\.png'.format('|'.join(image_types)))
        img_type = img_type_re.search(img_fn).group(1)
        if img_type == 'hr':
            img = img.resize(size, resample=Image.BICUBIC)
            img.save(img_path)
        elif img_type == 'lr':
            img = img.resize(size, resample=Image.NEAREST)
            img.save(img_path)


def main():
    for d in evaluate_dirs:
        collect_scores(d, size=(384,384))

if __name__ == '__main__':
    main()
