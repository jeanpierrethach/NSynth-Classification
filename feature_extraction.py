import numpy as np
import librosa

def feature_extract(file, n_mfcc, n_mels, fmax):
    """
    Define function that takes in a file an returns features in an array
    """
    
    # get wave representation
    y, sr = librosa.load(file)
        
    # determine if instrument is harmonic or percussive by comparing means
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    if np.mean(y_harmonic) > np.mean(y_percussive):
        harmonic = 1
    else:
        harmonic = 0
        
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # temporal averaging
    mfcc = np.mean(mfcc, axis=1)
    
    # get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)  
    # temporally average spectrogram
    spectrogram = np.mean(spectrogram, axis=1)
    
    # compute chroma energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    # temporally average chroma
    chroma = np.mean(chroma, axis=1)
    
    # compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = np.mean(contrast, axis=1)
    
    return [harmonic, mfcc, spectrogram, chroma, contrast]

def feature_extract_flatten(file, n_mfcc, n_mels, fmax):
    # get wave representation
    y, sr = librosa.load(file)
        
    # determine if instrument is harmonic or percussive by comparing means
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    if np.mean(y_harmonic) > np.mean(y_percussive):
        harmonic = 1
    else:
        harmonic = 0
        
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)  
    
    # compute chroma energy
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    
    # compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    return [harmonic, mfcc.flatten(), spectrogram.flatten(), chroma.flatten(), contrast.flatten()]

def extract_mel_spectrogram(file, n_mels, hop_length, fmax):
    # get wave representation
    y, sr = librosa.load(file)

    # get the mel-scaled spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, fmax=fmax)

    return spectrogram