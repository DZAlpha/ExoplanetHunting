import pandas as pd
import numpy as np
from dataread import *
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RF


class RandomForestFramework:
    
    def __init__(self,df,  number_of_estimators = 100, criterion = 'gini', max_depth = None, 
                  min_samples_split = 2, min_samples_leaf=1,min_weight_fraction_leaf = 0.0, max_features = 'auto', 
                  max_samples = None, bootstrap = True):
            
            ''' 
            The only argument we have to pass is df, bootstrap is allowed as it creates less correlated trees
            
            '''
            self.df = df
            self.number_of_estimators = number_of_estimators
            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split= min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_samples = max_samples
            self.bootstrap = bootstrap
            self.RandomForest = RF(number_of_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf,
                                  min_weight_fraction_leaf = min_weight_fraction_leaf, max_features =max_features, bootstrap = bootstrap)      
    
    
    def data_from_df(self, df, targets_name = 'LABEL', targets_from_index = True):
        '''
        Obtaining classes and samples from dataframe
        
        '''
        if targets_from_index == True:
            y = df.index.get_level_values(targets_name).to_numpy()
            X = df.reset_index(drop=True).to_numpy()
        else:
            y = df[target_name].to_numpy()
            X = df.drop(target_name).to_numpy()
        return X, y
        
    def fit(self, targets_name = 'LABEL', targets_from_index = True):
        '''
        Getting data from our dataframe and fitting it
            targets_from_index == True, because we'll likely have our labels in some kind of multiindex
        '''
        
        X, y = self.data_from_df(self.df, targets_name = targets_name, targets_from_index = targets_from_index)
        
        self.RandomForest.fit(X,y)
        
        
    def predict_df(self, df, targets_name = 'LABEL', targets_from_index = True):
        X, y = self.data_from_df(df, targets_name = targets_name, targets_from_index = targets_from_index)
        return self.predict(X)
        
    def predict(self, samples):
        return self.RandomForest.predict(samples)
    
    def get_errors(self, samples, targets, sample_weight = None):
        score = self.RandomForest.score(samples, targets, sample_weight = sample_weight)
        self.accuracy = score
        self.error = 1.00 - self.accuracy
        print("Forest of {} trees has {} accuracy, {} error. {} samples were classified correctly".format(self.number_of_estimators, 
                                                                                                   self.accuracy, self.error,
                                                                                                  int(self.accuracy * len(targets))))
    def draw_tree(self, n = None, target_name = 'LABEL'):

        # Draw random tree if not specified 
        if n==None or n >= self.number_of_estimators:
            n = np.random.randint(0, self.number_of_estimators)
        
        # Getting feature names   
        feature_names = self.df.columns.to_numpy()
        
        # Plotting tree
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
        tree.plot_tree(self.RandomForest.estimators_[n],
                       feature_names = feature_names, 
                       filled = True);
        