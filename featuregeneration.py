import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import signal
from dataread import preprocess, get_train_df
from BollingerBand import bollinger_band
from fourier import FourierTransform

# def get_spikes_feature(train_df,
#                        threshold_type = 'mean', 
#                        threshold_multiplier = 1,
#                        threshold_value = 1):
#     '''
#     Calculates number of spikes in discrete Fourier transform calculated from train_df using scipy.signal.find_peaks
#     Ignores first 10 values from fft because of a spike there that couldn't be removed.
#     Args:
#     train_df: dataframe to calculate Fourier transform from
#     threshold_type: 'mean', 'std' or 'number'. For 'mean'/'std' the threshold used in find_peaks will be respectively mean/standard deviation
#                     of absolute value of Fourier transform. For 'number' the threshold will be a real number passed 
#                     as threshold_value
#     threshold_multiplier: real number by which the threshold will be multiplied
#     threhsold_value: threshold base value used if threshold_type = 'regular'
#     '''
    
#     def get_number_of_spikes(row):
#         fft = np.fft.fft(scipy.signal.detrend(row))[10:]
#         if threshold_type == 'mean':
#             threshold = np.mean(np.absolute(fft))
#         elif threshold_type == 'std':
#             threshold = np.std(np.absolute(fft))
#         else:
#             threshold = threshold_value
#         threshold *= threshold_multiplier
#         return scipy.signal.find_peaks(np.absolute(fft)[:int(len(fft)/2)], threshold = threshold)[0].shape
    
#     spikes = np.array(train_df.apply(lambda row: get_number_of_spikes(row), axis = 1))
#     return np.array(list(map(lambda x : x[0], spikes)))

def get_df(normalized_df = None, standarized_df = None, dft = None):
    '''
    Returns dataframe with features and proper label.
    Current features: Standard Deviation, Variance, Mean
    Args:
    normalized_df: normalized dataframe with source values. 
            By deafult it's preprocess_normalize(get_train_df()). Used to generate most of the features
    standarized_df: standarized dataframe with source values.
            By deafult it's preprocess_standarize(get_train_df()). Used in features calculated from Fourier transformation.
    std_dst: distance for threshold argument for finding peaks in fourier transform. Used in get_number_of_spikes
    '''
    if normalized_df is None:
        normalized_df = preprocess(get_train_df())
    if standarized_df is None:
        standarized_df = preprocess(get_train_df(), preprocess='standarize')
    features = pd.DataFrame()
    features['Label'] = normalized_df.index.get_level_values(1)
    features['Standard deviation'] = np.array(normalized_df.apply(lambda row: row.std(), axis = 1))
    features['Variance'] = np.array(normalized_df.apply(lambda row: row.var(), axis = 1))
    features['Mean'] = np.array(normalized_df.apply(lambda row: row.mean(), axis = 1))
    features['Outliers'] = np.array(self.df.apply(lambda row: n_outliers(row), axis=1))
    if dft is not None:
        features['Fourier std'] = dft.std
        features['Fourier mean'] = dft.mean
        features['Fourier spikes'] = dft.spikes
        features['Fourier outliers from BBand'] = fourier_outliers(dft.fourier_df, right_index = int(dft.fourier_df.shape[1]/2))
        
    return features


def fourier_outliers(fourier_df, right_index, span = 20, k = 3.5):
    return np.array(fourier_df.apply(lambda row: len(row[:right_index][bollinger_band(row[:right_index], span = span, k = k)[2]]), axis = 1))

def n_outliers(row, span = 20, k = 3):
    return len(row[bollinger_band(row, span = span, k = k)[3]])    
    
class Features:
    def __init__(self, df, standarized_df):
        self.df = df
        self.standarized_df = standarized_df
        self.dft = FourierTransform(standarized_df, detrend = False)
        
        self.features = pd.DataFrame()
        self.features['Label'] = df.index.get_level_values(1)
        self.features['Standard deviation'] = np.array(self.df.apply(lambda row: row.std(), axis = 1))
        self.features['Variance'] = np.array(self.df.apply(lambda row: row.var(), axis = 1))
        self.features['Mean'] = np.array(self.df.apply(lambda row: row.mean(), axis = 1))
        self.features['Outliers'] = np.array(self.df.apply(lambda row: n_outliers(row), axis=1))
        
        self.features['Fourier std'] = self.dft.std
        self.features['Fourier mean'] = self.dft.mean
        self.features['Fourier spikes'] = self.dft.spikes
        self.features['Fourier outliers from BBand'] = fourier_outliers(self.dft.fourier_df, right_index = int(self.dft.fourier_df.shape[1]/2))
        
    def add_feature(self, fun, name, stand=False):
        if stand:
            self.features[name] = np.array(self.standarized_df.apply(lambda row: fun(row), axis=1))
        else: 
            self.features[name] = np.array(self.df.apply(lambda row: fun(row), axis=1))
    
    def drop_feature(self, name):
        self.features = self.features.drop(columns=[name])
                 