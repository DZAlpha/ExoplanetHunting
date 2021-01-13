import numpy as np
import pandas as pd

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def standarize(series):
    return (series - np.mean(series)) / np.std(series)

def add_indexes(df):
    ID = np.arange(len(df))
    label = df.index.values
    multi_index = list(zip(ID, label))
    index = pd.MultiIndex.from_tuples(multi_index, names = ["ID", "LABEL"])
    return df.set_index(index)

def preprocess(df, preprocess = 'normalize'):
    '''
    Preprocesses the dataframe according to the preprocess arg and adds indexes.
    Args:
        df: dataframe to preprocess
        preprocess: 'normalize' or 'standarize' 
    '''
    df = add_indexes(df)

    if preprocess == 'normalize':
        return df.apply(normalize, axis = 1)
    elif preprocess == 'standarize':
        return df.apply(standarize, axis = 1)
    else:
        return df

def get_train_df():
    '''
    Reads local csv file with training data
    '''
    return pd.read_csv('exoTrain.csv', index_col = 0)

def get_test_df():
    return pd.read_csv('exoTest.csv', index_col = 0)
