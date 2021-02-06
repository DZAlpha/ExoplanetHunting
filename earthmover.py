from dataread import *
import numpy as np
import pandas as pd
from BollingerBand import *
from scipy.stats import wasserstein_distance



def earth_for_row(series, outliers_function = bollinger_band):

    if outliers_function == bollinger_band:
        _, _, upper_outliers, lower_outliers, _ = bollinger_band(series)
        all_outliers = np.hstack([np.where(upper_outliers), np.where(lower_outliers)])[0]
    
        if np.size(all_outliers) == 0:
            return -1
        if np.size(all_outliers) == 1:
            return -1
        
    mini=np.sort(all_outliers)[0]
    fixed = np.sort(all_outliers) - mini
    
    length = np.size(fixed)
    maxi = np.flip(np.sort(fixed))[0]

    step = maxi/(length-1)

    perfect_outliers = np.arange(length) * step
    
    return wasserstein_distance(fixed, perfect_outliers)


def earth_mover_distance(df):
    results = df.apply(earth_for_row, axis=1)
    return results.to_numpy()


