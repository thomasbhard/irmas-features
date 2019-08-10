import os

import numpy as np
from python_speech_features import mfcc

from utils import calculate_nfft, get_files

# Global Variables
IRMAS_PATH = 'C:\\Users\\thoma\\Documents\\_STUDIUM\\5. Semester\\Fachvertiefung Software\\IRMAS-TrainingData'

SAMPLERATE = 44100
WINLEN = 0.025 # window length in seconds
WINSTEP = WINLEN # time between to windows e.g if WINSTEP=WINLEN there is now overlap
SEQLEN = 0 # sequence length, relevant for extracting features with sequencial information

WINLEN_SAMP = round(WINLEN*SAMPLERATE) # window length in samples
NFFT = calculate_nfft(SAMPLERATE, winlen=WINLEN)

# ---------------------------------------------------------------------------------------------


# Files

instruments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio']
num_files_per_inst = 10 # if none, all existing files are taken

filenames = []

for inst in instruments:
    filenames.extend(get_files(IRMAS_PATH, inst=inst, num=num_files_per_inst))

# ---------------------------------------------------------------------------------------------


def extract_features_window(s):
    """Extract features for one window

    :param s: 1-D array of the signal

    :return: returns feature vector for one window
    """

    assert np.shape(s)[0] == WINLEN_SAMP

    # MFCC 
    mfcc_features = mfcc(s, samplerate=SAMPLERATE, winlen=WINLEN, winstep=WINSTEP, nfft=NFFT)
    features = mfcc_features

    # Stat
    features = np.append(features, np.mean(s))
    features = np.append(features, np.mean(s))

    return features


        

if __name__ == "__main__":


    s = np.zeros(WINLEN_SAMP)

    features = extract_features_window(s)

    print(features)

    
