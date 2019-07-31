import os
import glob

import csv

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from visualisation import corrplot


PLOT = False

os.chdir('audio')
filenames = glob.glob('*.wav')

max_len = 0

for filename in filenames:
    y, sr = librosa.load(filename, sr=44100)

    max_len = max([len(y), max_len])

print(max_len)


df = pd.DataFrame()

for filename in filenames:
    y, sr = librosa.load(filename, sr=44100)

    padding = np.zeros(max_len - len(y))

    y = np.append(y, padding)

    features = librosa.feature.mfcc(y=y, sr=sr)
    features = np.ndarray.flatten(features, order='F')

    df[filename] = features[:20]

print(df.head())
corr = df.corr()

if PLOT:
    corrplot(corr)
    plt.show()

print(corr.head())

columns = list(corr.columns)

print(columns[0])
print(columns[1])


# corr_mean = np.mean(corr.to_numpy())
corr_mean = 0.8
print('Mean of correlation: ' + str(corr_mean))

with open('correlations.csv', mode='w') as corr_file:
    writer = csv.writer(corr_file, delimiter=',')
    writer.writerow(['Source', 'Target', 'Weight'])
    cnt = 0
    for i in range(len(columns) - 1):
        for j in range(i+1, len(columns)):
            curr_corr = corr.iloc[i,j]
            if curr_corr > corr_mean:
                writer.writerow([columns[i], columns[j], curr_corr])
                cnt += 1

print(str(cnt) + ' Entries')

            



