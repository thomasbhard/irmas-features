# irmas-features
Preprocessing of the IRMAS Dataset including feature extraction and feature selection



## Installation

There is no requirements.txt yet, so just install everything python tells you to - sorry for that.

---


## Setup Global Variables

All the global variables are stored as a json file in the tables folder with the same name as the actual table. HINT: In order for humans to be able to read the json in your IDE I recommend to use some sort of formatting tool.
There is no function to set the varibales from a config file yet. If you choose to randomize your files you wont be able to reproduce the exact same result.

### Collecting files from the Dataset.
#### Specify global path to the IRMAS-Dataset:
```python
IRMAS_PATH = 'C:\\Users\\pathtoirmas\\IRMAS-TrainingData'
```


#### The number of files used from each istrument. This defines how many features you will get.

Each file is about 3 seconds long which amounts to about 132 300 samples per file.

* Using the create_dataframe_slice function the number of features can be estimated as such:

    num_features = num_files_per_inst * 1 323 000 / slice_len

    **E.g using 10 files per instrument and a slice length of 1024 reuslts in 12900 
    features**

* Using the create_dataframe_domain function, the number of features depends on the WINLEN and WINSTEP aswell as the number of files per instrument
    
    **E.g using 10 files per instrument and a window length of 0.025 seconds without overlap results in 12100 features.**

```python
num_files_per_inst = 3
```
#### Shuffle data
If you don't use all the files you might consider to shuffle your files in order to get a randomly distributed set of files.
```python
rand = True
```

#### Options for filtering and ignoring certain files:
* Use filt if you only want files CONTAINING the string specified in filt.
```python
filt = 'jaz_blu'
```
* Use ignore if you want to ignore all the files containing one or more strings from the list.
```python
ignore = ['jaz_blu', 'pop_roc']
```


### Outputfile

As the feature extraction can be timeconsuming you probably want to safe featuretables as a csv. Use the following variable as the filename in order to avoid problems in other modules.

```python
ouputfile = os.path.join(os.path.abspath(__file__), '..', '..', 'tables', 'name_of_your_featuretable.csv')
```

---

## Feature Extraction
```python
mode = 'domain' # can be domain, slice or sequence
```

### Domain Features
Calculating MFCCs and statistical features.

Parameters concerning domain features
```python
winlen = 0.025  # window length in seconds
winstep = winlen  # time between to windows e.g if WINSTEP=WINLEN there is now overlap

```
---

### Sliceing the files
If you want to use the raw data as the input. 

```python
slice_len = 1024
```
--- 

### Sequence for LSTM
Not supported yet.







