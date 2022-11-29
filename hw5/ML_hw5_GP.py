import numpy as np

def loadData(filepath):
    X = np.zeros((34))
    Y = np.zeros((34))
    line_num = 0

    with open(filepath) as f:
        for line in f.readlines():
            x, y = line.split(' ')
            X[line_num] = x
            Y[line_num] = y
            line_num += 1
    
    return X, Y


X, Y = loadData("./data/input.data")
