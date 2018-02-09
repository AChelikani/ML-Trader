import csv

########################################
# Data Format
# Date,Open,High,Low,Close,Volume,Name
########################################

class DataReader(object):
    def __init__(self, filename):
        self.data = {}
        self.read_data(filename)

    def read_data(self, filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            # Skip header row
            next(reader, None)
            for row in reader:
                try:
                    cleaned_row = [row[0]] + list(map(float, row[1:-1]))
                    ticker = row[-1]
                    if ticker not in self.data:
                        self.data[ticker] = cleaned_row
                    else:
                        self.data[ticker].append(cleaned_row)
                except:
                    continue

            print("Stocks read: "+ str(len(self.data)))

if __name__ == "__main__":
    dr = DataReader("data/all_stocks_1yr.csv")
