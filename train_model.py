import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

from utils import maybe_make_directory, write_metadata

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
	parser.add_argument('--epochs', type=int,
						default=30,
						help='Number of epochs. (default: %(default)s)')
	parser.add_argument('--batch_size', type=int,
						default=32,
						help='Batch size. (default: %(default)s)')
	parser.add_argument('--learning_rate', type=float,
						default=1e-4,
						help='Learning rate parameter for the Adam optimizer. (default: %(default)s)')
	parser.add_argument('--meta_name', type=str,
                        default='meta_data',
                        help='Configuration file output')

	args = parser.parse_args()
	maybe_make_directory(args.input_dir)
	maybe_make_directory(args.model_dir)
	maybe_make_directory(args.graph_dir)
	return args

args = parse_args()

X_train = np.load(os.path.join(args.input_dir, "train_spectro.npy"))
y_train = np.load(os.path.join(args.input_dir, "train_spectro_labels.npy"))

X_valid = np.load(os.path.join(args.input_dir, "valid_spectro.npy"))
y_valid = np.load(os.path.join(args.input_dir, "valid_spectro_labels.npy"))

print(X_train.shape)
print(X_valid.shape)

num_classes = 11

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
				 input_shape=(X_train.shape[1:])))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())

opt = keras.optimizers.Adam(lr=args.learning_rate)

model.compile(loss='categorical_crossentropy',
			  optimizer=opt,
			  metrics=['accuracy'])

early_stopping = [keras.callbacks.EarlyStopping(monitor='val_loss',
							  min_delta=0,
							  patience=2,
							  verbose=0, mode='auto')]

history = model.fit(X_train, y_train,
		  batch_size=args.batch_size,
		  epochs=args.epochs,
		  validation_data=(X_valid, y_valid),
		  shuffle=True, callbacks=early_stopping, verbose=1)


timestr = time.strftime("%Y%m%d-%H%M%S")

model_name = 'keras_nsynth_' + timestr + '.h5'
model_path = os.path.join(args.model_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

write_metadata(os.path.join(args.model_dir, args.meta_name + '_' + timestr + '_.txt'), model_name, args)

print(history.history.keys())

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig(os.path.join(args.graph_dir, 'model_accuracy_'+ timestr + '.png'))
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig(os.path.join(args.graph_dir, 'model_loss_ '+ timestr + '.png'))
plt.close()
