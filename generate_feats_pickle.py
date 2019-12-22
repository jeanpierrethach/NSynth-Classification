import os
import numpy as np
import pandas as pd
import pickle
import argparse

import librosa

from utils import instrument_code, maybe_make_directory
from feature_extraction import feature_extract, feature_extract_flatten

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='train', choices=['train', 'valid', 'test'],
                        help='Dataset to generate pickle of features. (default: %(default)s)')
    parser.add_argument('--output_dir', type=str,
                        default='./out',
                        help='Directory path to the outputs. (default: %(default)s)')
    parser.add_argument('--temp_avg', action='store_true',
                        help='Boolean flag activating temporal averaging of features. (default: False)')
    parser.add_argument('--sample', action='store_true',
                        help='Boolean flag activating sampling of training set')
    parser.add_argument('--n_samples', type=int,
                        default=5000,
                        help='Number of samples to take from each instrument family. (default: %(default)s)')
    parser.add_argument('--n_mfcc', type=int,
                        default=13,
                        help='Number of MFCCs to return. (default: %(default)s)')
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
    if args.dataset == "train":
        path = "./nsynth-train.jsonwav.tar/nsynth-train"
    elif args.dataset == "valid":
        path = "./nsynth-valid.jsonwav.tar/nsynth-valid"
    elif args.dataset == "test":
        path = "./nsynth-test.jsonwav.tar/nsynth-test"
    
    maybe_make_directory(args.output_dir)

    return args, path

args, path = parse_args()

os.chdir("/mnt/d/IFT6390_NSynth_data")

AUDIO_FILEPATH = "./audio/"


def mfcc(df_features):
    mfcc = pd.DataFrame(df_features.mfcc.values.tolist(),
                                index=df_features.index)
    return mfcc.add_prefix('mfcc_')

def spectrogram(df_features):
    spectrogram = pd.DataFrame(df_features.spectro.values.tolist(),
                                    index=df_features.index)
    return spectrogram.add_prefix('spectro_')

def chroma(df_features):
    chroma = pd.DataFrame(df_features.chroma.values.tolist(),
                                    index=df_features.index)
    return chroma.add_prefix('chroma_')

def contrast(df_features):
    contrast = pd.DataFrame(df_features.contrast.values.tolist(),
                                    index=df_features.index)
    return contrast.add_prefix('contrast_')

class NSynthFeatureExtractor(object):
    def __init__(self, path, temporal_averaging=False):
        self.DEFAULT_FEATURES = [mfcc, spectrogram, chroma, contrast]
        self.temporal_averaging = temporal_averaging

    def extract(self, filenames):
        dict_feats = {}
        for filename in filenames:
            if self.temporal_averaging:
                features = feature_extract(os.path.join(path, AUDIO_FILEPATH, filename + '.wav'), n_mfcc=args.n_mfcc, n_mels=args.n_mels, fmax=args.fmax)
                dict_feats[filename] = features
            else:
                features = feature_extract_flatten(os.path.join(path, AUDIO_FILEPATH, filename + '.wav'), n_mfcc=args.n_mfcc, n_mels=args.n_mels, fmax=args.fmax)
                dict_feats[filename] = features                

        df_features = pd.DataFrame.from_dict(dict_feats, orient='index',
                                            columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])
        return self._build_features(df_features)

    def _build_features(self, df_features):
        df_default_feats = [feature(df_features) for feature in self.DEFAULT_FEATURES]

        df_features = df_features.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)

        df_features = pd.concat([df_features] + df_default_feats,
                                axis=1, join='inner')

        targets = []
        for name in df_features.index.tolist():
            targets.append(instrument_code(name))
        df_features['targets'] = targets

        return df_features

    def dump(self, df_features):
        if self.temporal_averaging:
            with open(os.path.join(args.output_dir, 'df_features_' + args.dataset + '.pickle'), 'wb') as f:
                pickle.dump(df_features, f)
        else:
            with open(os.path.join(args.output_dir, 'df_features_' + args.dataset + '_all_features.pickle'), 'wb') as f:
                pickle.dump(df_features, f)


nsynthfe = NSynthFeatureExtractor(path, temporal_averaging=args.temp_avg)

df_examples_json = pd.read_json(path_or_buf=os.path.join(path, 'examples.json'), orient='index')


if args.dataset == "train" and args.sample:
    df_examples_json = df_examples_json.groupby('instrument_family', as_index=False,
                                group_keys=False).apply(lambda df: df.sample(args.n_samples, random_state=0))

filenames_examples = df_examples_json.index.tolist()

with open(os.path.join(args.output_dir, 'filenames_' + args.dataset + '.pickle'), 'wb') as f:
    pickle.dump(filenames_examples, f)

df_features = nsynthfe.extract(filenames_examples)
nsynthfe.dump(df_features)

print(df_features.shape)