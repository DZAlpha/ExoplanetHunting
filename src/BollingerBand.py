import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.dataread import * 


def MA(series,span):
    '''Moving Average'''
    return series.rolling(span).mean()

def std(series, span):
    return series.rolling(span).std()


def bollinger_band(series,span = 20, k = 3):
    moving_a = MA(series, span)
    std_ = std(series, span)
    upper_bound = moving_a + k*std_
    lower_bound = moving_a - k*std_
    values_ = series
    
    upper_outliers = values_ > upper_bound
    lower_outliers = values_ < lower_bound
    
    return upper_bound, lower_bound, upper_outliers, lower_outliers, moving_a