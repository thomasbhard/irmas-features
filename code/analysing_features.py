import os

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tsfresh.feature_selection.significance_tests import target_real_feature_real_test
from tsfresh.feature_selection.relevance import calculate_relevance_table

from visualisation import corrplot


# read feature table
feature_table = 'testfeatures.csv'
df = pd.read_csv(os.path.join(os.path.abspath(__file__), '..', '..', 'tables', feature_table), index_col=0)
print(df.head())

X = df.iloc[:,:-1]
print(X.std())

def plot_corr_matrix():
    corr = df.iloc[:,:-1].corr()

    plt.figure(figsize=(10, 10))
    corrplot(corr)

    plt.show()

def significane_test():
    X = df.iloc[:,:-1]
    dummy = np.random.rand(len(df.index))
    X['dummy'] = dummy


    y = df.iloc[:,-1]

    le = LabelEncoder()
    le.fit(['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio'])
    y_t = le.transform(y)
    y_t = pd.Series(y_t)

    

    print(calculate_relevance_table(X, y_t, ml_task='classification'))






if __name__ == "__main__":
    pass
    # plot_corr_matrix()
    # significane_test()


