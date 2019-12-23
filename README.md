# NSynth Classification


Datasets : [NSynth](https://magenta.tensorflow.org/datasets/nsynth#files)

## Setup
#### Dependencies:
* python3, [tensorflow](https://github.com/tensorflow/tensorflow), [keras](https://github.com/keras-team/keras), [scikit-learn](https://github.com/scikit-learn/scikit-learn), [Pillow](https://github.com/python-pillow/Pillow), [librosa](https://github.com/librosa/librosa), etc.

#### To install the dependencies:
```
pip3 install -r requirements.txt
```

## Usage
### Basic Usage

1. Run the commands **in order** with specific arguments for the `train`, `valid` and `test` sets.

## Extracting features and generating a pickle file 
```
python3 generate_feats_pickle.py [-h] <arguments> 
```
*Example*:
```
python3 generate_feats_pickle.py --dataset train --data_path /nsynth-train.jsonwav.tar/nsynth-train/ --sample --n_samples 1000
```

#### Arguments
* `--dataset` : Dataset to generate pickle of features. *Choices*: `{train,valid,test}` **(required)** *Default*: `None`
* `--data_path`: Directory path to the datasets. **(required)** *Default*: `None`
* `--output_dir`: Directory path to the outputs. *Default*: `./out`
* `--temp_avg`: Boolean flag activating temporal averaging of features. *Default*: `False`
* `--sample`: Boolean flag activating sampling of training set.  *Default*: `False`
* `--n_samples`: Number of samples to take from each instrument family.  *Default*: `5000`.
* `--n_mfcc`: Number of MFCCs to return.  *Default*: `13`
* `--n_mels`: Number of Mel bands to generate.  *Default*: `128`
* `--fmax`: Highest frequency (in Hz).  *Default*: `8000`
* `--hop_length`: Number of samples between successive frames.  *Default*: `1024`

## Generate melspectrograms images
```
python3 generate_melspectrograms.py [-h] <arguments> 
```
*Example*:
```
python3 generate_melspectrograms.py --dataset train --data_path /nsynth-train.jsonwav.tar/nsynth-train/
```

#### Arguments
* `--dataset` : Dataset to load the filenames pickle. *Choices*: `{train,valid,test}` **(required)** *Default*: `None`
* `--data_path`: Directory path to the datasets. **(required)** *Default*: `None`
* `--input_dir`: Directory path to the pickle files. *Default*: `./out`
* `--output_dir`: Directory path to the melspectrograms output. *Default*: `./melspec`
* `--n_mels`: Number of Mel bands to generate.  *Default*: `128`
* `--fmax`: Highest frequency (in Hz).  *Default*: `8000`
* `--hop_length`: Number of samples between successive frames.  *Default*: `1024`

## Create the melspectrograms dataset
```
python3 create_dataset_melspectrograms.py [-h] <arguments> 
```
*Example*:
```
python3 create_dataset_melspectrograms.py --dataset train --verbose --resize --size 32,32
```

#### Arguments
* `--dataset` : Dataset to load melspectrograms and save as a npy. *Choices*: `{train,valid,test}` **(required)** *Default*: `None`
* `--resize`: Boolean flag activating resize of melspectrograms. *Default*: `False`
* `--size`: Size must be height,width.  *Default*: `32,32`
* `--input_dir`: Directory path to the melspectrograms. *Default*: `./melspec`
* `--output_dir`: Directory path to the outputs. *Default*: `./out`

* `--verbose`: Boolean flag activating console prints.  *Default*: `False`


## Model training
```
python3 train_model.py [-h] <arguments> 
```
*Example*:
```
python3 train_model.py --epochs 20 --batch_size 64
```

#### Arguments
* `--size`: Size must be height,width.  *Default*: `32,32`
* `--input_dir`: Directory path to the npy files. *Default*: `./out`
* `--output_dir`: Directory path to the model output. *Default*: `./models`
* `--graph_dir`: Directory path to the graphs output. *Default*: `./graphs`
* `--epochs`: Number of epochs.  *Default*: `30`
* `--batch_size`: Batch size.  *Default*: `32`
* `--learning_rate`:  Learning rate parameter for the Adam optimizer. *Default*: `1e-4`
* `--meta_name`: Configuration file output.  *Default*: `meta_data`


## Model evaluation
```
python3 eval_model.py [-h] <arguments> 
```
*Example*:
```
python3 eval_model.py --model_path models/keras_nsynth_20191223-104657.h5
```

#### Arguments
* `--input_dir`: Directory path to the npy files. *Default*: `./out`
* `--graph_dir`: Directory path to the graphs output. *Default*: `./graphs`
* `--model_path`:  Path of the model. **(required)**
* `--normalize`: Boolean flag activating normalization of the confusion matrix.  *Default*: `False`


# Authors 
Thach Jean-Pierre *- University of Montreal*

Li Lin *- University of Montreal*

Yabo Ling *- University of Montreal*

