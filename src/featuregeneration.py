import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import signal
from src.dataread import preprocess, get_train_df
from src.BollingerBand import bollinger_band
from src.fourier import FourierTransform


def fourier_outliers(fourier_df, right_index, span = 20, k = 3.5):
    return np.array(fourier_df.apply(lambda row: len(row[:right_index][bollinger_band(row[:right_index], span = span, k = k)[2]]), axis = 1))


def n_outliers(row, span = 20, k = 3):
    return len(row[bollinger_band(row, span = span, k = k)[3]]) 


def out_idx(row):
    return np.diff(np.where(bollinger_band(row, span=20, k=2.5)[3])[0])


def d_std(series):
    if len(series)>1:
        return series.std()
    return 3197.


def d_mean(series):
    if len(series)>1:
        return series.mean()
    return 0.
   
    
class Features:
    def __init__(self, df, standarized_df):
        self.df = df
        self.standarized_df = standarized_df
        self.dft = FourierTransform(standarized_df, detrend = False)
        
        self.features = pd.DataFrame()
        self.features['Label'] = df.index.get_level_values(1)
        self.features['Flux: STD'] = np.array(self.df.apply(lambda row: row.std(), axis = 1))
        self.features['Flux: variance'] = np.array(self.df.apply(lambda row: row.var(), axis = 1))
        self.features['Flux: mean'] = np.array(self.df.apply(lambda row: row.mean(), axis = 1))
        #bollinger band
        self.features['Flux: No. BB Outliers'] = np.array(self.df.apply(lambda row: n_outliers(row), axis=1))
        idx = self.df.apply(lambda row: out_idx(row), axis=1)
        self.features['Flux: STD of BB outliers'] = np.array(idx.apply(lambda row: d_std(row)))
        self.features['Flux: mean of BB outliers'] = np.array(idx.apply(lambda row: d_mean(row)))
        self.features['Fourier: STD'] = self.dft.std
        self.features['Fourier: mean'] = self.dft.mean
        #self.features['Fourier outliers #1'] = self.dft.spikes
        self.features['Fourier: No. BB outliers'] = fourier_outliers(self.dft.fourier_df, right_index = int(self.dft.fourier_df.shape[1]/2))
        
    def add_feature(self, fun, name, stand=False):
        if stand:
            self.features[name] = np.array(self.standarized_df.apply(lambda row: fun(row), axis=1))
        else: 
            self.features[name] = np.array(self.df.apply(lambda row: fun(row), axis=1))
    
    def drop_feature(self, name):
        self.features = self.features.drop(columns=[name])
                 