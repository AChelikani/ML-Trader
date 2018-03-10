from sklearn import linear_model, svm, metrics
import data_reader

class Trainer(object):
    def __init__(self, data, outputs, sklearn_obj):
        self.data = data
        self.outputs = outputs
        self.clf = sklearn_obj

    def train(self, stock):
        stock_data = self.data[stock]
        stock_outputs = self.outputs[stock]

        split_percent = .80
        split_pos = int(len(stock_data) * split_percent)
        print split_pos

        train_x = stock_data[:split_pos]
        self.test_x = stock_data[split_pos:]
        train_y = stock_outputs[:split_pos]
        self.test_y = stock_outputs[split_pos:]

        self.clf.fit(train_x, train_y)

    def predict(self):
        return self.clf.predict(self.test_x)

    def evaluate(self):
        return self.predict() - self.test_y

    def score(self):
        return self.clf.score(self.test_x, self.test_y)

if __name__ == "__main__":
    dr = data_reader.DataReader("data/all_stocks_5yr.csv")
    dr.gen_outputs()
    trainer = Trainer(dr.data, dr.outputs, linear_model.LinearRegression())
    for stock in dr.data:
        trainer.train(stock)
        print stock, trainer.score()
