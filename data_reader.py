import csv

########################################
# Data Format
# Date,Open,High,Low,Close,Volume,Name
########################################

class DataReader(object):
    def __init__(self, filename):
        self.data = {}
        self.outputs = {}
        self.read_data(filename)

    def read_data(self, filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            # Skip header row
            next(reader, None)
            for row in reader:
                try:
                    cleaned_row = [row[0]] + list(map(float, row[1:-1]))
                    cleaned_row = cleaned_row[1:]
                    ticker = row[-1]
                    if ticker not in self.data:
                        self.data[ticker] = [cleaned_row]
                    else:
                        self.data[ticker].append(cleaned_row)
                except:
                    continue

    def gen_outputs(self):
        for stock in self.data:
            outputs = []
            states = self.data[stock]
            for x in range(1,len(states)):
                outputs.append(states[x][0])
            self.data[stock] = states[:-1]
            self.outputs[stock] = outputs


if __name__ == "__main__":
    dr = DataReader("data/all_stocks_1yr.csv")
