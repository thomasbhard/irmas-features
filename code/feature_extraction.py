import os
import time

import numpy as np
import pandas as pd
import scipy.stats
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.utils.np_utils import to_categorical
from python_speech_features import mfcc
from python_speech_features.sigproc import framesig

from utils import calculate_nfft, get_files, get_label, printProgressBar


times = []

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
label_encoder = LabelEncoder()
label_encoder.fit(instruments)
num_files_per_inst = 10 # if none, all existing files are taken

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

    t = time.time()

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

    dur = time.time() - t

    times.append(dur)



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
    """Creates a dataframe with the specified features 

    The specified features are extracted from the specified files and scaled using the StandardScaler.

    :return: pandas dataframe with the corresponding label as the last column
    """

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



def get_slices(slice_len=512):
    """Slice audiofiles into slices of size 'slice_len'

    In addition to slicing, each slice is normalized between -1 and 1.

    :param slice_len: number of samples of one slice
    
    :return: returns a tuple containg:
                -a 2-D array with the shape (slice_len, num_slices) 
                 where num_slices is defined by slice_len and the number of files specified in the 
                 global variables section
                -a 2-D array with the shape (num_labels, num_slices) containg one-hot-encoded labels
    """

    
    features = None
    labels = None

    num_files = len(filenames)
    progress = 0

    printProgressBar(progress, num_files, prefix='Progress', suffix='Complete', length=50)

    for f in filenames:

        # read file
        _, data = wav.read(f)
        data = data[:,0]

        num_slices = len(data) // slice_len
        assert num_slices > 0, 'slice_len is to big'
        num_samples = num_slices * slice_len

        new_features = np.array(np.split(data[:num_samples], num_slices), dtype=np.float32)
        

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

    for feature in features:
        feature /= np.max(np.abs(feature))

    return features, labels


def create_dataframe_slices(slice_len=256):
    """Create a dateframe with features from the get_slices function.

    In addtion to extracting the features, the labels are one hot encoded using
    the label encoder and to_categorical from keras utils.

    :param slice_len: number of samples in one slice
    :return: pandas dataframe with featues and labels 
    """

    features, labels = get_slices(slice_len=slice_len)

    labels_enc = label_encoder.transform(labels)
    labels_one_hot = to_categorical(labels_enc)

    df_features = pd.DataFrame(features)
    df_features.columns = ['Sample ' + str(i) for i in range(slice_len)]

    df_labels = pd.DataFrame(labels_one_hot)
    df_labels.columns = instruments

    assert len(df_features.index) == len(df_labels.index)

    df = pd.concat([df_features, df_labels], axis=1, join='inner')
    
    return df


if __name__ == "__main__":
    # example usage feature extraction
    df_features = create_dataframe()
    df_features.to_csv(outputfile)

    #example usage slicing
    df_slices = create_dataframe_slices(slice_len=1024)
    df_slices.to_csv(outputfile)



