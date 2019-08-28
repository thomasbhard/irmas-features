"""Analysing features and comparing classification results

Plotting correlation matrix, using pca, and classifing using a simple NN.

This script was used to explore the IRMAS Dataset and to try out different methods 
for visualizing and classifying. This is now split into feature_extraction 
and analysing_features.
"""

import os
import csv

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from visualisation import corrplot


PLOT = True
TRAIN = False
TRAIN_ON_PCA = False

# read features
feature_table = 'features_gel_pia_mfcc.csv'
columns = ['MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12', 'MFCC13', 'Label']
df = pd.read_csv(os.path.join(os.path.abspath(__file__), '..', '..', 'tables', feature_table), header=None, names=columns)
print(df.head())

corr = df.iloc[:,:13].corr()

if PLOT:
    # plot correlation matrix
    plt.figure(figsize=(10, 10))
    corrplot(corr)

    plt.show()

if PLOT:
    # plot spring map
    corr_mean = 0.2
    corr_file = os.path.join(os.path.abspath(__file__), '..', '..', 'tables', 'correlations.csv')
    with open(corr_file, mode='w') as corr_file:
        writer = csv.writer(corr_file, delimiter=',')
        writer.writerow(['Source', 'Target', 'Weight'])
        cnt = 0
        for i in range(len(columns[0:-1]) - 1):
            for j in range(i+1, len(columns[0:-1])):
                curr_corr = corr.iloc[i,j]
                curr_corr = abs(curr_corr)
                if curr_corr > corr_mean:
                    writer.writerow([columns[i], columns[j], curr_corr])
                    cnt += 1


# Scale
X = df.iloc[:,:13]
Y = df.iloc[:,13]

X_scaled = StandardScaler().fit_transform(X)

# PCA
pca = PCA(n_components=13)
X_pca = pca.fit_transform(X_scaled)


# Scatterplot using 2 principal components
principalDf = pd.DataFrame(data=X_pca[:,:2], columns=['PCA1', 'PCA2'])
finalDf = pd.concat([principalDf, df[['Label']]], axis=1)

if PLOT:
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0, 1]
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Label'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PCA1']
                , finalDf.loc[indicesToKeep, 'PCA2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()

    plt.show()

# Split into training and testing data
if TRAIN_ON_PCA:
    x_train, x_test, y_train, y_test = train_test_split(X_pca, Y)
else:
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y)

# one hot encoding for training labels
y_train = np_utils.to_categorical(y_train)

# Define Neural Net

model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

if TRAIN:
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2, verbose=1)

    # Evaluation
    plt.plot(history.epoch, history.history['loss'])
    plt.show()

    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    score = accuracy_score(y_test, y_pred)

    print('Score using principal components: ' + str(score))

