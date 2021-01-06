import numpy as np
import pandas as pd

# url_path ='https://raw.githubusercontent.com/DZAlpha/ExoplanetHunting/main/exoTest.csv'
# df = pd.read_csv(url_path, index_col=0)

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

def preprocess(df):
  '''
  Normalizes the dataframe and adds indexes
  '''
  df = add_indexes(df)
  return df.apply(normalize, axis = 1)

def get_train_df():
  '''
  Reads local csv file with training data
  '''
  return pd.read_csv('exoTrain.csv', index_col = 0)

def get_test_df():
  return pd.read_csv('exoTest.csv', index_col = 0)



# ID = np.arange(570)
# label = df.index.values

# multi_index = list(zip(ID,label))
# index = pd.MultiIndex.from_tuples(multi_index, names=["ID", "LABEL"])
# df = df.set_index(index)

# # Normalized and standarized data frames
# normalized_df = df.apply(normalize,axis=1)
# standarized_df = df.apply(standarize, axis=1)