import numpy as np
import pandas as pd

url_test ='https://raw.githubusercontent.com/DZAlpha/ExoplanetHunting/main/exoTest.csv'
df = pd.read_csv(url_path, index_col=0)


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def standarize(series):
    return (series - np.mean(series)) / np.std(series)


ID = np.arange(570)
label = df.index.values

multi_index = list(zip(ID,label))
index = pd.MultiIndex.from_tuples(multi_index, names=["ID", "LABEL"])
df = df.set_index(index)

# Normalized and standarized data frames
normalized_df = df.apply(normalize,axis=1)
standarized_df = df.apply(standarize, axis=1)