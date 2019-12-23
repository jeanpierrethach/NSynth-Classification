import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

import glob
from PIL import Image

from keras.utils import to_categorical

from utils import instrument_code, maybe_make_directory

def _size(s):
    try:
        return tuple(map(int, s.split(',')))
    except:
        raise argparse.ArgumentTypeError("Size must be height,width")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        required=True, choices=['train', 'valid', 'test'],
                        help='Dataset to load melspectrograms and save as a npy. (required) (default: %(default)s)')
    parser.add_argument('--resize', action='store_true',
                        help='Boolean flag activating resize of melspectrograms (default: False)')
    parser.add_argument('--size', type=_size, nargs=1,
                        default='32,32',
                        help='Size must be height,width (default: (%(default)s))')
    parser.add_argument('--input_dir', type=str,
                        default='./melspec',
                        help='Directory path to the melspectrograms. (default: %(default)s)')
    parser.add_argument('--output_dir', type=str,
                        default='./out',
                        help='Directory path to the outputs. (default: %(default)s)')
    parser.add_argument('--verbose', action='store_true',
                        help='Boolean flag activating console prints (default: False)')
    args = parser.parse_args()

    maybe_make_directory(args.output_dir)

    return args

args = parse_args()

def load_image(filename, resize=False):
    img = Image.open(filename)
    img.load()
    if args.resize:
        img.thumbnail(args.size[0], Image.ANTIALIAS)
        if args.verbose:
            print(f"Resizing melspectrogram to : {img.size} with PIL.IMAGE.ANTIALIAS")
    data = np.asarray(img, dtype=np.float32)
    return data

def read_melspectrograms():
    mel_spectrograms = []
    labels = []
    melspec_path = os.path.join(args.input_dir, args.dataset, "*.png")
    for im_path in glob.glob(melspec_path):
        if args.verbose:
            print(im_path)
        im = load_image(im_path, resize=True)
        mel_spectrograms.append(im)
        labels.append(instrument_code(im_path.split(os.sep)[3]))
    return mel_spectrograms, labels

mel_spectrograms, labels = read_melspectrograms()

X = np.array(mel_spectrograms)
y = to_categorical(np.array(labels))

X_output_path = os.path.join(args.output_dir, f'{args.dataset}_melspectro.npy')
y_output_path = os.path.join(args.output_dir, f'{args.dataset}_melspectro_labels.npy')
np.save(X_output_path, X)
np.save(y_output_path, y)

print(f'Melspectrograms examples saved at {X_output_path}')
print(f'Melspectrograms labels saved at {y_output_path}')