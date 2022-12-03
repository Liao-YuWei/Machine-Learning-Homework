import numpy as np
from libsvm.svmutil import *
import sys

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

def get_best_option(X_train, Y_train, option, optimal_option, optimal_accuarcy):
    accuracy = svm_train(Y_train, X_train, option)
    if accuracy > optimal_accuarcy:
        return option, accuracy
    else:
        return optimal_option, optimal_accuarcy

def grid_search(X_train, Y_train, kernel_type):
    cost = [0.001, 0.01, 0.1, 1, 10, 100]
    optimal_option = f'-s 0 -v 3 -q'
    optimal_accuarcy = 0

    if kernel_type == 0:    #linear
        print('\nlinear kernel:')
        for c in cost:
            option = f'-s 0 -t {kernel_type} -c {c} -q -v 3'
            print(option)
            optimal_option, optimal_accuarcy = get_best_option(X_train, Y_train, option, optimal_option, optimal_accuarcy)
    elif kernel_type == 1:  #polynomial
        gamma = [0.0001, 1/784, 0.01, 0.1, 1, 10]
        coefficient = [-10, -5, 0, 5, 10]
        degree = [1, 2, 3, 4]

        print('\npolynomial kernel:')
        for c in cost:
            for d in degree:
                for g in gamma:
                    for r in coefficient:
                        option = f'-s 0 -t {kernel_type} -c {c} -d {d} -g {g} -r {r} -q -v 3'
                        print(option)
                        optimal_option, optimal_accuarcy = get_best_option(X_train, Y_train, option, optimal_option, optimal_accuarcy)
    elif kernel_type == 2:  #RBF
        gamma = [0.0001, 1/784, 0.01, 0.1, 1, 10]

        print('\nRBF kernel:')
        for c in cost:
            for g in gamma:
                option = f'-s 0 -t {kernel_type} -c {c} -g {g} -q -v 3'
                print(option)
                optimal_option, optimal_accuarcy = get_best_option(X_train, Y_train, option, optimal_option, optimal_accuarcy)
    elif kernel_type == 3:  #sigmoid
        gamma = [0.0001, 1/784, 0.01, 0.1, 1, 10]
        coefficient = [-10, -5, 0, 5, 10]

        print('\nsigmoid kernel:')
        for c in cost:
            for g in gamma:
                for r in coefficient:
                    option = f'-s 0 -t {kernel_type} -c {c} -g {g} -r {r} -q -v 3'
                    print(option)
                    optimal_option, optimal_accuarcy = get_best_option(X_train, Y_train, option, optimal_option, optimal_accuarcy)

    optimal_option = optimal_option[:-5]    #remove -v
    print(optimal_accuarcy)
    print(optimal_option)

    return optimal_option

def linear_kernel(u, v):
    return u @ v.T

def RBFkernel(u, v):
    gamma = 1 / 784
    dist = np.sum(u ** 2, axis=1).reshape(-1, 1) + np.sum(v ** 2, axis=1) - 2 * u @ v.T

    return np.exp(-gamma * dist)


part = int(input('Please choose a part to run(1, 2, 3): '))
X_train, Y_train, X_test, Y_test = load_data('./data/X_train.csv', './data/Y_train.csv', './data/X_test.csv', './data/Y_test.csv')
train_num = X_train.shape[0]
test_num = X_test.shape[0]

if part == 1:
    for kernel_type in range(3):
        model = svm_train(Y_train, X_train, f'-t {kernel_type} -q')
        result = svm_predict(Y_test, X_test, model)
elif part == 2:
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open('SVM_part2.txt', 'w') as f:
        sys.stdout = f
        for kernel_type in range(4):
            optimal_option = grid_search(X_train, Y_train, kernel_type)
            model = svm_train(Y_train, X_train, optimal_option)
            result = svm_predict(Y_test, X_test, model)
        sys.stdout = original_stdout # Reset the standard output to its original value
elif part == 3:
    train_kernel = linear_kernel(X_train, X_train) + RBFkernel(X_train, X_train)
    test_kernel = linear_kernel(X_train, X_test) + RBFkernel(X_test, X_test)

    # Add index in front of kernel
    train_kernel = np.hstack((np.arange(1,train_num + 1).reshape(-1, 1), train_kernel))
    test_kernel = np.hstack((np.arange(1,test_num + 1).reshape(-1, 1), test_kernel))

    model = svm_train(Y_train, train_kernel, '-t 4 -q')
    result = svm_predict(Y_test, test_kernel, model)
else:
    print('Wrong input!')