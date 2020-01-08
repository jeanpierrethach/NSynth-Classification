import os
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def instrument_code(filename):
    """
    Function that takes in a filename and returns instrument based on naming convention
    """
    class_names = {'bass': 0, 
                'brass': 1,
                'flute': 2,
                'guitar':3,
                'keyboard':4,
                'mallet':5,
                'organ':6,
                'reed':7,
                'string':8,
                'synth':9,
                'vocal':10
    }
    class_file = filename.split('_')[0]
    return class_names[class_file]

def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    np.set_printoptions(threshold=np.inf)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.043)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Note: Fix for matplotlib 3.1.1 https://github.com/matplotlib/matplotlib/issues/14675
    bt, tp = ax.get_ylim()
    ax.set_ylim(bt + 0.5, tp - 0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    return ax

def write_metadata(path, model_name, args):
    with open(path, 'w') as f:
        f.write(f'model_name: {model_name}\n')
        f.write(f'epochs: {args.epochs}\n')
        f.write(f'batch_size: {args.batch_size}\n')
        f.write(f'learning_rate: {args.learning_rate}\n')