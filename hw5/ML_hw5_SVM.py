import numpy as np
from libsvm.svmutil import *

def load_data(x_train_filepath, y_train_filepath, x_test_filepath, y_test_filepath):
    X_train = np.empty((0,784))
    Y_train = []
    X_test = np.empty((0,784))
    Y_test = []

    print('Loading training images...')
    with open(x_train_filepath) as f:
        for line in f.readlines():
            image = line.split(',')
            image = np.array(image).astype(float)
            X_train = np.vstack((X_train, image))
    f.close()

    print('Loading training labels...')
    with open(y_train_filepath) as f:
        for line in f.readlines():
            label = float(line)
            Y_train.append(label)
    f.close()

    print('Loading test images...')
    with open(x_test_filepath) as f:
        for line in f.readlines():
            image = line.split(',')
            image = np.array(image).astype(float)
            X_test = np.vstack((X_test, image))
    f.close()

    print('Loading test labels...')
    with open(y_test_filepath) as f:
        for line in f.readlines():
            label = float(line)
            Y_test.append(label)
    f.close()

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_data('./data/X_train.csv', './data/Y_train.csv', './data/X_test.csv', './data/Y_test.csv')
for kernel_type in range(3):
    model = svm_train(Y_train, X_train, f'-t {kernel_type} -q')
    result = svm_predict(Y_test, X_test, model)