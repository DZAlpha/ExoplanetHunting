import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_row(df: pd.DataFrame, row_index):
    '''
    Plots info about one row of a dataframe.
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
    
    degrees = [1, 4, 10]
    for degree in degrees:
        plt.plot(xs, np.poly1d(np.polyfit(xs, df.iloc[row_index], degree))(xs), label = f'Degree {degree}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

