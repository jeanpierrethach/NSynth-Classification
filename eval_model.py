import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

import keras
import tensorflow as tf
from keras.models import load_model

from utils import plot_confusion_matrix, maybe_make_directory

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', type=str,
						default='./out',
						help='Directory path to the npy files. (default: %(default)s)')
	parser.add_argument('--model_dir', type=str,
						default='./models',
						help='Directory path to the model output. (default: %(default)s)')
	parser.add_argument('--graph_dir', type=str,
						default='./graphs',
						help='Directory path to the graphs output. (default: %(default)s)')
	parser.add_argument('--model_name', type=str,
						required=True,
						help='Name of the model. (required)')
	parser.add_argument('--normalize', action='store_true',
                        help='Boolean flag activating normalization of the confusion matrix')

	args = parser.parse_args()
	maybe_make_directory(args.input_dir)
	maybe_make_directory(args.model_dir)
	maybe_make_directory(args.graph_dir)
	return args

args = parse_args()

class_names = np.array(['bass', 'brass', 'flute', 'guitar', 
			 'keyboard', 'mallet', 'organ', 'reed', 
			 'string', 'synth', 'vocal'])

X_test = np.load(os.path.join(args.input_dir, "test_spectro.npy"))
y_test = np.load(os.path.join(args.input_dir, "test_spectro_labels.npy"))

model = load_model(os.path.join(args.model_dir, args.model_name))

scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

y_preds = np.argmax(model.predict(X_test), axis=1)
y_test = np.argmax(y_test, axis=1)

plot_confusion_matrix(y_test, y_preds, classes=class_names, normalize=args.normalize,
					  title='Normalized confusion matrix for CNN')
plt.savefig(os.path.join(args.graph_dir, f'cnn_normalized_{args.normalize}_{time.strftime("%Y%m%d-%H%M%S")}.png'))