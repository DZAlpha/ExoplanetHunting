import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from BollingerBand import *

def plot_fourier_transform(ft):
    n = len(ft)
    plt.plot(np.arange(n), ft)

def plot_row(df: pd.DataFrame, row_index, degrees = [1, 4, 10], title = ""):
    '''
    Plots info about one row of a dataframe.
    Args:
    df: dataframe 
    row_index: row to plot info about
    degrees: degrees of polynomials to compute
    '''
    label = df.index[row_index][1]
    if label == 2:
        print("There is an exoplanet")
    else:
        print("There's no exoplanet")
        
    xs = np.arange(len(df.iloc[row_index]))
    plt.figure(figsize = (10, 5))
    plt.ylabel('Flux')
    plt.xlabel('Time')
    plt.title(title)
    plt.plot(xs, df.iloc[row_index])

    for degree in degrees:
        plt.plot(xs, np.poly1d(np.polyfit(xs, df.iloc[row_index], degree))(xs), label = f'Degree {degree}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plot_histograms(df):
    '''
    Plots histograms for exoplanet dataframe with columns: 'Mean', 'Standard deviation', 'Variance'
    Args:
    df: df to get values from
    '''

    def plot_histogram(column, title):
        plt.figure(figsize = (10, 5))
        sns.distplot(column)
        plt.title(title)
        plt.show()

    for column in df.columns:
        if column != 'Label':
            print(column)
            plot_histogram(df[column], title = 'All data')
            plot_histogram(df[df['Label'] == 2][column], title = 'Exoplanet exists')
            plot_histogram(df[df['Label'] == 1][column], title = 'Exoplanet doesn\'t exist')
            
def plot_band(series, span = 20, k=3, scatter = True, figsize = (15, 5)):
    ''' Plots Bollinger band for series'''

    upper,lower,upper_outliers,lower_outliers, moving_a = bollinger_band(series, span, k)
    plt.figure(figsize = figsize)
    
    x_ = np.arange(1,len(series) + 1)

    plt.plot(x_, upper, c='green')
    plt.plot(x_, lower, c='blue')
    plt.plot(x_, moving_a, c='black')
    plt.scatter(np.where(upper_outliers), series[upper_outliers], c = 'red', s=70)
    plt.scatter(np.where(lower_outliers), series[lower_outliers], c='red',s=70)
    if scatter:
        plt.scatter(x_, series, c='orange', s=20)
    else:
        plt.plot(x_, series)
