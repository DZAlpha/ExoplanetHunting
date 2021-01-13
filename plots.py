import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def printsomething():
  print("AAAA")

def plot_fourier_transform(ft):
    n = len(ft)
    plt.plot(np.arange(n), ft)

def plot_row(df: pd.DataFrame, row_index, degrees = [1, 4, 10]):
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

    plt.plot(xs, df.iloc[row_index])

    for degree in degrees:
        plt.plot(xs, np.poly1d(np.polyfit(xs, df.iloc[row_index], degree))(xs), label = f'Degree {degree}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plot_histograms(df):
    '''
    Plots histograms for exoplanet dataframe with columns: 'Mean', 'Standard deviation', 'Variance'
    Args:
    df: 
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