import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from plots import plot_row

class FourierTransform:
    def __init__(self, df, detrend = True, left_index = 10):
        '''
        Calculates the df where each row is filled with absolute values of Discrete Fourier transform
        and saves it to self.fourier_df. 
        Args:
        df: dataframe to calculate Discrete Fourier transform form. FFT is applied to each row. 
        detrend: bool deciding if original df rows should be detrended using scipy.signal.detrend
        left_index: index of the first left element that will be considered in calculating std and mean. 
        '''
        self.original_df = df.copy()
        self.fourier_df = df.copy()
        for i in range(len(df)):
            if detrend:
                self.fourier_df.iloc[i] = np.absolute(np.fft.fft(scipy.signal.detrend(self.fourier_df.iloc[i])))
            else:
                self.fourier_df.iloc[i] = np.absolute(np.fft.fft(self.fourier_df.iloc[i]))
        self.calculate_std(left_index = left_index)
        self.calculate_mean(left_index = left_index)
        self.calculate_no_spikes()
                
    def calculate_no_spikes(self):
        '''
        Calculates number of spikes in each row of self.fourier_df and saves it to self.no_spikes.
        If the function had already been called, the function only returns the value it calculated the last time. 
        Returns:
        Ndarray filled with number of spikes.
        '''
        self.spikes = np.array(self.fourier_df.apply(lambda row: self.__spikes_in_series(row), axis = 1))
    
    def calculate_std(self, left_index = 10):
        '''
        Calculates std of every row of self.fourier_df and saves it to self.std
        Args:
        left_index: index of the first left element that will be considered. 
        '''
        right_index = int(self.fourier_df.shape[1]/2)
        self.std = np.array(self.fourier_df.apply(lambda row: np.std(row[left_index:right_index]), axis = 1))
        
    def calculate_mean(self, left_index = 10):
        '''
        Calculates mean of every row of self.fourier_df and saves it to self.mean
        Args:
        left_index: index of the first left element that will be considered. 
        '''
        right_index = int(self.fourier_df.shape[1]/2)
        self.mean = np.array(self.fourier_df.apply(lambda row: np.mean(row[left_index:right_index]), axis = 1))
    
    def __spikes_in_series(self, series, threshold_type = 'mean', threshold_multiplier = 1.9, left_index = 10):
        '''
        Method used to calculate number of spikes from pandas.Series
        Args:
        series: values from which spikes will be calculated
        threshold_type: 'mean' or 'std'
        threshold_multiplier: real number by which the threshold will be multiplied
        left_index: indexes of series smaller than left_index will be ignored
        Returns:
        Number of spikes
        '''
        right_index = int(len(series)/2)
        series = series[left_index:right_index]
        if threshold_type == 'mean':
            threshold = np.mean(series)
        else:
            threshold = np.std(series)
        threshold *= threshold_multiplier
        return len(scipy.signal.find_peaks(series, threshold = threshold)[0])
        
        
    def plot_row(self, row_index, figsize = (12, 6), plot_source_values = False, left_index = 0):
        if row_index < 0 or row_index > len(self.fourier_df):
            print("row_index out of bounds")
            return 
        message = "There is an exoplanet" if self.fourier_df.index[row_index][1] == 2 else "There is no exoplanet"
        plt.figure(figsize = figsize)
        plt.xlabel('Frequency')
        right_index = int(self.fourier_df.shape[1] / 2)
        plt.plot(np.arange(left_index, right_index), self.fourier_df.iloc[row_index][left_index:right_index])
        if plot_source_values:
            plot_row(self.original_df, row_index)
        else:
            print(message)
        print("std = ", np.std(self.fourier_df.iloc[row_index][left_index:right_index]))