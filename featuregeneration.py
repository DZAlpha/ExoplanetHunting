import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import signal
from dataread import *


def get_spikes_feature(train_df,
                       threshold_type = 'mean', 
                       threshold_multiplier = 1,
                       threshold_value = 1):
    '''
    Calculates number of spikes in discrete Fourier transform calculated from train_df using scipy.signal.find_peaks
    Ignores first 10 values from fft because of a spike there that couldn't be removed.
    Args:
    train_df: dataframe to calculate Fourier transform from
    threshold_type: 'mean', 'std' or 'number'. For 'mean'/'std' the threshold used in find_peaks will be respectively mean/standard deviation
                    of absolute value of Fourier transform. For 'number' the threshold will be a real number passed 
                    as threshold_value
    threshold_multiplier: real number by which the threshold will be multiplied
    threhsold_value: threshold base value used if threshold_type = 'regular'
    '''
    
    def get_number_of_spikes(row):
        fft = np.fft.fft(scipy.signal.detrend(row))[10:]
        if threshold_type == 'mean':
            threshold = np.mean(np.absolute(fft))
        elif threshold_type == 'std':
            threshold = np.std(np.absolute(fft))
        else:
            threshold = threshold_value
        threshold *= threshold_multiplier
        return scipy.signal.find_peaks(np.absolute(fft)[:int(len(fft)/2)], threshold = threshold)[0].shape
    
    spikes = np.array(train_df.apply(lambda row: get_number_of_spikes(row), axis = 1))
    return np.array(list(map(lambda x : x[0], spikes)))

def get_df(normalized_df = None, standarized_df = None):
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
        normalized_df = preprocess_normalize(get_train_df())
    if standarized_df is None:
        standarized_df = preprocess_standarize(get_train_df())
    features = pd.DataFrame()
    features['Label'] = normalized_df.index.get_level_values(1)
    features['Standard deviation'] = np.array(normalized_df.apply(lambda row: row.std(), axis = 1))
    features['Variance'] = np.array(normalized_df.apply(lambda row: row.var(), axis = 1))
    features['Mean'] = np.array(normalized_df.apply(lambda row: row.mean(), axis = 1))
    features['Spikes (from mean, multiplier = 1.9)'] = get_spikes_feature(standarized_df, threshold_type = 'mean', threshold_multiplier = 1.9)
    return features
