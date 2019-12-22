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
        height, width = map(int, s.split(','))
        return height, width
    except:
        raise argparse.ArgumentTypeError("Size must be height,width")

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str,
						default='train', choices=['train', 'valid', 'test'],
						help='Dataset to generate pickle of features. (default: %(default)s)')
	parser.add_argument('--resize', action='store_true',
                        help='Boolean flag activating resize of melspectrograms')
	parser.add_argument('--size', type=_size, nargs=2,
						default='32,32',
                        help='Size must be height,width (default: (%(default)s))')
	parser.add_argument('--output_dir', type=str,
						default='./out',
						help='Directory path to the outputs. (default: %(default)s)')
	parser.add_argument('--max_size', type=int,
						default=1920,
						help='Maximum width or height of the input images. (default: %(default)s)')  
	parser.add_argument('--input_dir', type=str,
						default='./melspec',
						help='Directory path to the melspectrograms. (default: %(default)s)')
	args = parser.parse_args()

	maybe_make_directory(args.output_dir)

	return args

args = parse_args()

def load_image(filename, resize=False):
	img = Image.open(filename)
	img.load()
	if resize:
		img.thumbnail(args.size, Image.ANTIALIAS)
		print(f"Resizing melspectrograms to : {args.size}")
	data = np.asarray(img, dtype=np.float32)
	return data

def read_melspectrograms():
	mel_spectrograms = []
	labels = []
	melspec_path = os.path.join(args.input_dir, args.dataset, "*.png")
	for im_path in glob.glob(melspec_path):
		im = load_image(im_path, resize=True)
		mel_spectrograms.append(im)
		labels.append(instrument_code(im_path))
	return mel_spectrograms, labels

mel_spectrograms, labels = read_melspectrograms()

X = np.array(mel_spectrograms)
y = to_categorical(np.array(labels))

np.save(os.path.join(args.output_dir, args.dataset + "_melspectro.npy"), X)
np.save(os.path.join(args.output_dir, args.dataset + "_melspectro_labels.npy"), y)