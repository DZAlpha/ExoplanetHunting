import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataread import *


def get_df(train_df = None):
    '''
    Returns dataframe with features and proper label.
    Current features: Standard Deviation, Variance, Mean
    Args:
    train_df: dataframe with source values. If nothing is passed, it will be dataread.preprocess(get_train_df())
    '''
    if train_df is None:
        train_df = preprocess(get_train_df())
    features = pd.DataFrame()
    features['Label'] = train_df.index.get_level_values(1)
    features['Standard deviation'] = np.array(train_df.apply(lambda row: row.std(), axis = 1))
    features['Variance'] = np.array(train_df.apply(lambda row: row.var(), axis = 1))
    features['Mean'] = np.array(train_df.apply(lambda row: row.mean(), axis = 1))
    return features
