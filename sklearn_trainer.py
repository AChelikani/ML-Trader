import numpy as np
import pandas as pd
from sklearn import linear_model, svm, metrics

class SKLearnTrainer(object):
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
    classifiers = [
        linear_model.SGDRegressor(),
        linear_model.BayesianRidge(),
        linear_model.LassoLars(),
        linear_model.ARDRegression(),
        linear_model.PassiveAggressiveRegressor(),
        linear_model.TheilSenRegressor(),
        linear_model.LinearRegression(),
        svm.SVR()]

    for classifier in classifiers:
        trainer = SKLearnTrainer(classifier, 'data/data_stocks.csv')
        x_train, y_train, x_test, y_test = trainer.split_data(.80)
        trainer.train(x_train, y_train)
        print trainer.get_score(x_test, y_test)
