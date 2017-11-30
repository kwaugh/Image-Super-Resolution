import argparse
import os
import re
import subprocess

one_dir = 'calibrate_ones/'
five_dir = 'calibrate_fives/'

def load_file_list(path, regx='\.png'):
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    return return_list

def show_images(path, viewer_app='eog'):
    img_list = load_file_list(path, regx='\.png')

    viewers = []
    for img_fn in img_list:
        img_path = os.path.join(path, img_fn)
        viewers.append(subprocess.Popen([viewer_app, img_path]))

    input('Press enter to close the images.')

    for v in viewers:
        v.terminate()
        v.kill()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', default='eog')
    args = parser.parse_args()

    input("Thanks for volunteering to rate image quality for Keivaun and Paul's vision project! Press enter to continue.")
    input("In this experiment, you'll be asked to rate the image quality of images on a scale of 1 to 5. Press enter to continue.")
    print("These images have an image quality rating of 1.")
    show_images(one_dir, viewer_app=args.viewer)
    print("These images have an image quality rating of 5.")
    show_images(five_dir, viewer_app=args.viewer)
    print("Run `python3 calibrate.py` to get started. Thanks again!")

if __name__ == '__main__':
    main()
