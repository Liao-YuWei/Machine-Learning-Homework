import numpy as np

def load_data(x_train_filepath, y_train_filepath, x_test_filepath, y_test_filepath):
    X_train = np.empty((0,784))
    Y_train = np.empty((0,1))
    X_test = np.empty((0,784))
    Y_test = np.empty((0,1))

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
            Y_train = np.vstack((Y_train, label))
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
            Y_test = np.vstack((Y_test, label))
    f.close()

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_data('./data/X_train.csv', './data/Y_train.csv', './data/X_test.csv', './data/Y_test.csv')