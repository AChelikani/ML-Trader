import gmpy2
import math
import time
import numpy
import csv
from random import *
import numpy as np
from sklearn import neural_network
import pandas as pd
# Need to standardize the data for the MLP
from sklearn.preprocessing import StandardScaler

# MLP Regressor on Market Data
class SKLearnMLP(object):
    def __init__(self, sklearn_obj, data_file):
        self.clf = sklearn_obj
        self.data_file = data_file
        self.clean_data(self.data_file)

    def clean_data(self, data_file):
        data = pd.read_csv(data_file)

        data = data.drop(['DATE'], 1)

        self.n = data.shape[0]
        self.p = data.shape[1]

        self.data = data.values

    def split_data(self, split_percent):
        train_start = 0
        train_end = int(np.floor(split_percent*self.n))
        test_start = train_end + 1
        test_end = self.n
        data_train = self.data[np.arange(train_start, train_end), :]
        data_test = self.data[np.arange(test_start, test_end), :]
        x_train = data_train[:, 1:]
        y_train = data_train[:, 0]
        x_test = data_test[:, 1:]
        y_test = data_test[:, 0]

        return x_train, y_train, x_test, y_test

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x_test):
        return self.clf.predict(x_test)

    def evaluate(self, predictions, y_test):
        return y_test - predictions

    def get_score(self, x_test, y_test):
        score = self.clf.score(x_test, y_test)
        return score

if __name__ == "__main__":
    classifiers = []
    #alpha_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    neural_layouts = [(5, 5, 5, 5, 5), (25, ), (5, 4, 3, 2, 1), (10, 5, 3, 2, 5)]
    for layout in neural_layouts:
        classifiers.append(neural_network.MLPRegressor(max_iter = 10000, hidden_layer_sizes=layout, early_stopping = True))
    i = 0
    print classifiers
    for classifier in classifiers:
        trainer = SKLearnMLP(classifier, 'data_stocks.csv')
        x_train, y_train, x_test, y_test = trainer.split_data(.80)
        divisor = []
#        for i in range(len(x_train)):
#            divisor.append(sum(x_train[i])/y_train[i])
#        print np.std(divisor)
#        print sum(divisor)/len(divisor)        

        trainer.train(x_train, y_train)
        print i
        print trainer.get_score(x_test, y_test)
        i = i + 1
