# irmas-features
Preprocessing of the IRMAS Dataset including feature extraction and feature selection

## Installation

There is no requirements.txt yet, so just install everything python tells you to - sorry for that.

## Usage

### Setup Global Variables

#### Specify the path to the root folder of the IRMAS-TrainingData.
```python
IRMAS_PATH = 'C:\\Users\\pathtoirmas\\IRMAS-TrainingData'
```
---
#### The number of files used from each istrument. This defines how many features you will get.

Each file is about 3 seconds long which amounts to about 132 300 samples per file.

* Using the create_dataframe_slice function the number of features can be estimated as such:

    num_features = num_files_per_inst * 1 323 000 / slice_len

    **E.g using 10 files per instrument and a slice length of 1024 reuslts in 12900 
    features**

* Using the create_dataframe function, the number of features depends on the WINLEN and WINSTEP aswell as the number of files per instrument
    
    **E.g using 10 files per instrument and a window length of 0.025 seconds without overlap results in 12100 features.**

```python
num_files_per_inst = 3
```
---
#### Outputfile

As the feature extraction can be timeconsuming you probably want to safe featuretables as a csv. Use the following variable as the filename in order to avoid problems in other modules.

```python
ouputfile = os.path.join(os.path.abspath(__file__), '..', '..', 'tables', 'name_of_your_featuretable.csv')
```

### Calculating MFFC and Stat features
In order to calculate MFCCs and statistical features, use the create_dataframe function. The dataframe contains all the features and a column with the corresponding label as a string.

### Slice the files
If you want to use the raw data as the input use the create_dataframe_slice function. The slice length is an optinal parameter. **NOTE:** The labels are already onehotencoded!







