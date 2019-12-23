import os
import numpy as np
import pickle
import argparse

import librosa
import librosa.display
import matplotlib.pyplot as plt

from utils import maybe_make_directory
from feature_extraction import extract_mel_spectrogram

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str,
						required=True, choices=['train', 'valid', 'test'],
						help='Dataset to generate pickle of features. (required) (default: %(default)s)')
	parser.add_argument('--data_path', type=str,
						required=True,
						help='Directory path to the datasets. (required) (default: %(default)s)')
	parser.add_argument('--input_dir', type=str,
						default='./out',
						help='Directory path to the pickle files. (default: %(default)s)')
	parser.add_argument('--output_dir', type=str,
						default='./melspec',
						help='Directory path to the melspectrograms output. (default: %(default)s)')
	parser.add_argument('--n_mels', type=int,
						default=128,
						help='Number of Mel bands to generate. (default: %(default)s)')
	parser.add_argument('--fmax', type=int,
						default=8000,
						help='Highest frequency (in Hz). (default: %(default)s)')
	parser.add_argument('--hop_length', type=int,
						default=1024,
						help='Number of samples between successive frames. (default: %(default)s)')

	args = parser.parse_args()

	maybe_make_directory(args.input_dir)
	maybe_make_directory(os.path.join(args.output_dir, args.dataset))

	return args

args = parse_args()

DATASET_PATH = args.data_path
AUDIO_FILEPATH = "./audio/"

FILENAMES_PICKLE_PATH = os.path.join(args.input_dir, f'filenames_{args.dataset}.pickle')

with open(FILENAMES_PICKLE_PATH, 'rb') as f:
	filenames_examples = pickle.load(f)

for idx, filename in enumerate(filenames_examples):
	mel_spectrogram = extract_mel_spectrogram(os.path.join(DATASET_PATH, AUDIO_FILEPATH, filename + '.wav'), n_mels=args.n_mels, hop_length=args.hop_length, fmax=args.fmax)
	class_file = filename.split('_')[0]
	plt.figure(figsize=(10,4))
	librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), 
						 y_axis='mel', fmax=args.fmax, x_axis='time')
	plt.colorbar(format='%+2.0f dB')
	plt.title("Mel spectrogram for " + class_file)
	plt.savefig(os.path.join(args.output_dir, args.dataset, f'{class_file}_mel_spectro_{str(idx)}.png'), bbox_inches='tight', transparent=True, pad_inches=0)
	plt.close()	
print(f'Mel spectrograms saved in {args.output_dir} directory')