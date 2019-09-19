import os
import time
import json

import numpy as np
import pandas as pd
import scipy.stats
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.utils.np_utils import to_categorical
from python_speech_features import mfcc
from python_speech_features.sigproc import framesig

from utils import calculate_nfft, get_files, get_label, printProgressBar


# DATASET
irmas_path = 'C:\\Users\\thoma\\Documents\\_STUDIUM\\5. Semester\\Fachvertiefung Software\\IRMAS-TrainingData'
samplerate = 44100

# FILES
instruments = ['cel', 'cla', 'flu', 'gac',
               'gel', 'org', 'pia', 'sax', 'tru', 'vio']

num_files_per_inst = 3  # if none, all existing files are taken
rand = False
filt = ''
ignore = ['jaz_blu', 'pop_roc']


filenames = []

for inst in instruments:
    filenames.extend(get_files(irmas_path, inst=inst, filt=filt,
                               ignore=ignore, rand=rand, num=num_files_per_inst))

label_encoder = LabelEncoder()
label_encoder.fit(instruments)
# ---------------------------------------------------------------------------------------------

# FEATURE EXTRACTION
mode = 'slice'  # can be domain, slice or sequence

# DOMAIN
winlen = 0.025  # window length in seconds
winstep = winlen  # time between to windows e.g if WINSTEP=WINLEN there is now overlap
winlen_samp = round(winlen*samplerate)  # window length in samples
winlen_samp = round(winstep*samplerate)  # window step in samples
nfft = calculate_nfft(samplerate, winlen=winlen)

# SLICE
slice_len = 1048

# SEQUENCE
seqlen = 0  # sequence length, relevant for extracting features with sequencial information
# ---------------------------------------------------------------------------------------------

# OUTPUT
outputfile = os.path.join(os.path.abspath(
    __file__), '..', '..', 'tables', 'testfeatures_slice.csv')
# ---------------------------------------------------------------------------------------------

# Features
feature_names = ['MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10',
                 'MFCC11', 'MFCC12', 'MFCC13', 'Mean', 'Median', 'stdDeviation', 'q25', 'q75', 'iqr', 'skewness', 'kurtosis']
# ---------------------------------------------------------------------------------------------


def collect_settings():
    """Collects all the settings and saves them as a json file.

    The name is like outputfile + '_config.json'
    There is no function to read a config file yet.
    """
    print('Collecting settings...')

    abspath = os.path.abspath(outputfile)
    conf_file = os.path.splitext(abspath)[0] + '_config.json'

    data = {}
    data['irmas_path'] = irmas_path
    data['samplerate'] = samplerate
    data['instruments'] = instruments
    data['num_files_per_inst'] = num_files_per_inst
    data['rand'] = rand
    data['filt'] = filt
    data['ignore'] = ignore
    data['winlen'] = winlen
    data['winstep'] = winstep
    data['winlen_samp'] = winlen_samp
    data['winlen_samp'] = winlen_samp
    data['nfft'] = nfft
    data['slice_len'] = slice_len
    data['seqlen'] = seqlen
    data['outputfile'] = outputfile

    with open(conf_file, 'w') as fp:
        json.dump(data, fp)


# Domain Fucntions
def extract_features_window(s):
    """Extract features for one window

    :param s: 1-D array of the signal

    :return: returns feature vector for one window
    """

    assert np.shape(s)[0] == winlen_samp

    # MFCC
    mfcc_features = mfcc(s, samplerate=samplerate,
                         winlen=winlen, winstep=winstep, nfft=nfft)
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

    frames = framesig(data, winlen_samp, winlen_samp)

    features = None

    for frame in frames:
        if features is None:
            features = extract_features_window(frame)
        else:
            new_features = extract_features_window(frame)
            features = np.vstack((features, new_features))

    return features


def create_dataframe_domain():
    """Creates a dataframe with the specified features 

    The specified features are extracted from the specified files and scaled using the StandardScaler.

    :return: pandas dataframe with the corresponding label as the last column
    """

    features = None
    labels = None

    num_files = len(filenames)
    progress = 0

    printProgressBar(progress, num_files, prefix='Progress',
                     suffix='Complete', length=50)

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
        printProgressBar(progress, num_files, prefix='Progress',
                         suffix='Complete', length=50)

    # scale data
    features_scaled = StandardScaler().fit_transform(features)

    df_features = pd.DataFrame(features_scaled)
    df_features.columns = feature_names

    # df['Label'] = labels NOTE: Old version simple string versions instead of one hot

    labels_enc = label_encoder.transform(labels)
    labels_one_hot = to_categorical(labels_enc)


    df_labels = pd.DataFrame(labels_one_hot)
    df_labels.columns = instruments

    assert len(df_features.index) == len(df_labels.index)

    df = pd.concat([df_features, df_labels], axis=1, join='inner')

    print(df.head())

    return df

# Slicing Functions


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

    printProgressBar(progress, num_files, prefix='Progress',
                     suffix='Complete', length=50)

    for f in filenames:

        # read file
        _, data = wav.read(f)
        data = data[:, 0]

        num_slices = len(data) // slice_len
        assert num_slices > 0, 'slice_len is to big'
        num_samples = num_slices * slice_len

        new_features = np.array(
            np.split(data[:num_samples], num_slices), dtype=np.float16)

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
        printProgressBar(progress, num_files, prefix='Progress',
                         suffix='Complete', length=50)

    for feature in features:
        feature_max = np.max(np.abs(feature))
        if feature_max != 0:
            feature /= feature_max

    return features, labels


def create_dataframe_slice():
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

    print(df.head())

    return df

# Sequence Functions
# TODO: support for lstm networks


if __name__ == "__main__":

    if mode is 'domain':
        df = create_dataframe_domain()
    elif mode is 'slice':
        df = create_dataframe_slice()
    elif mode is 'sequence':
        print('Not implemented yet!')

    print('Saving csv...')
    df.to_csv(outputfile)
    collect_settings()
