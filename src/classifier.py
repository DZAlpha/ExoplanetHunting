import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Classifier:
    def __init__(self, model, n_estimators = 100, true_y = None):
        '''
        Args:
        model: object that implements methods:
        fit(X, y), predict(X)
        n_estimaros: number of estimators used by a model if a model uses more than one estimator (like a forest). 
        '''
        assert(model.fit)
        assert(model.predict)
        self.model = model(n_estimators = n_estimators)
        self.n_estimators = n_estimators
        self.predicts = None 
        self.true_y = true_y
        self.fitted = False
        self.tested = False
        
    def fit(self, X, y):
        self.model.fit(X, y)    
        self.fitted = True
        
    def fit_df(self, df, label_column = 'Label'):
        X, y = Classifier.df_to_ndarray(df, label_column)
        self.model.fit(X, y)
    
    def predict(self, X):
        if self.fitted:
            self.predicts = self.model.predict(X)
            self.tested = True
            return self.predicts
        
    def predict_df(self, df, label_column = 'Label'):
        X, y = Classifier.df_to_ndarray(df, label_column)
        self.predicts = self.model.predict(X)
        self.tested = True
        self.true_y = y
        return self.predicts
    
    def get_cfm(self):
        if self.tested:
            return confusion_matrix(self.true_y, self.predicts)
        
    def plot_cfm(self):
        cfm = self.get_cfm()
        plt.figure(figsize = (8, 6))
        sns.heatmap(cfm, cmap='Blues', annot=True, fmt='g')
        plt.show()

    def get_measures(self):
        '''
        Calculates and returns f_score, precision, recall, true negative rate
        '''
        if not self.tested:
            return
        cfm = self.get_cfm()
        tn, fp, fn, tp = cfm[0][0], cfm[0][1], cfm[1][0], cfm[1][1]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        true_negative_rate = tn / (tn + fp)
        precision, recall
        f_score = 2*((precision * recall)/(precision + recall))
        self.f_score, self.precision, self.recall, self.true_negative_rate = f_score, precision, recall, true_negative_rate
        return f_score, precision, recall, true_negative_rate
    
        
    def df_to_ndarray(df, label_column = 'Label'):
        '''
        Returns tuple (df as ndarray, label column as ndarray)
        '''
        return df.drop(columns = [label_column]).to_numpy(), df[label_column].to_numpy()
        