import os

import numpy as np
import pandas as pd
import scipy.stats
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
from python_speech_features import mfcc
from python_speech_features.sigproc import framesig

from utils import calculate_nfft, get_files, get_label, printProgressBar



# Global Variables
IRMAS_PATH = 'C:\\Users\\thoma\\Documents\\_STUDIUM\\5. Semester\\Fachvertiefung Software\\IRMAS-TrainingData'

SAMPLERATE = 44100
WINLEN = 0.025 # window length in seconds
WINSTEP = WINLEN # time between to windows e.g if WINSTEP=WINLEN there is now overlap
SEQLEN = 0 # sequence length, relevant for extracting features with sequencial information

WINLEN_SAMP = round(WINLEN*SAMPLERATE) # window length in samples
WINSTEP_SAMP = round(WINSTEP*SAMPLERATE) # window step in samples
NFFT = calculate_nfft(SAMPLERATE, winlen=WINLEN)

# ---------------------------------------------------------------------------------------------


# Files

instruments = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio']
num_files_per_inst = 3 # if none, all existing files are taken

filenames = []

for inst in instruments:
    filenames.extend(get_files(IRMAS_PATH, inst=inst, num=num_files_per_inst))

ouputfile = os.path.join(os.path.abspath(__file__), '..', '..', 'tables', 'testfeatures.csv')

# ---------------------------------------------------------------------------------------------

# Features

feature_names = ['MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Mean', 'Median', 'stdDeviation', 'q25', 'q75', 'iqr', 'skewness', 'kurtosis']

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
    mean = np.mean(s)
    features = np.append(features, mean)

    median = np.median(s)
    features = np.append(features, median)

    stdDeviation = np.std(s)
    features = np.append(features, stdDeviation)

    q25, q75 = np.percentile(s, [25, 75])
    features = np.append(features, q25)
    features = np.append(features, q75)

    iqr = q75 - q25
    features = np.append(features, iqr)

    skewness = scipy.stats.skew(s)
    features = np.append(features, skewness)

    kurtosis = scipy.stats.kurtosis(s)
    features = np.append(features, kurtosis)

    return features


def extract_features_file(filename):
    """Extract feature vectors for file

    Frames the file according to WINLEN and WINSTEP and extracts a featurevector for every frame.
    
    :param filename: filename of the IRMAS dataset

    :return: 2D-numpy array with the shape (numframes, numfeatures) NOTE: if numframes == 1 it returns a single featurevector

    """

    # read wav file
    data = wav.read(filename)
    data = data[1]
    data = data[:, 1]

    frames = framesig(data, WINLEN_SAMP, WINSTEP_SAMP)

    features = None

    for frame in frames:
        if features is None:
            features = extract_features_window(frame)
        else:
            new_features = extract_features_window(frame)
            features = np.vstack((features, new_features))
    
    return features


def create_dataframe():

    features = None
    labels = None

    num_files = len(filenames)
    progress = 0

    printProgressBar(progress, num_files, prefix='Progress', suffix='Complete', length=50)

    for f in filenames:

        new_features = extract_features_file(f)
        if features is None:
            features = new_features
        else:
            features = np.vstack((features, new_features))

        label = get_label(f)
        num_labels = np.shape(new_features)[0]
        new_labels = np.repeat(label, num_labels)

        if labels is None:
            labels = new_labels
        else:
            labels = np.append(labels, new_labels)

        progress += 1
        printProgressBar(progress, num_files, prefix='Progress', suffix='Complete', length=50)

    # scale data
    features_scaled = StandardScaler().fit_transform(features)
    

    df = pd.DataFrame(features_scaled)
    df.columns = feature_names

    df['Label'] = labels

    print(df.head())

    return df





if __name__ == "__main__":

    df = create_dataframe()
    df.to_csv(ouputfile)
    



    
