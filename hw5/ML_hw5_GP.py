import numpy as np
from numpy import matmul as mul
from numpy.linalg import inv
import matplotlib.pyplot as plt

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

def rational_qudartic_kernel(xa, xb, alpha, lengthscale, variance):
    k = (1 + ((xa-xb) ** 2) / (2 * alpha * lengthscale**2)) ** (-alpha) * variance
    return k

def create_covariance_matrix(X, beta, alpha, lengthscale, kernel_variance):
    num_data = X.shape[0]
    C = np.zeros((num_data, num_data))
    for row in range(num_data):
        for col in range(num_data):
            C[row][col] = rational_qudartic_kernel(X[row], X[col], alpha, lengthscale, kernel_variance)
            if row == col:
                C[row][col] += 1 / beta
    return C

def GP_predict(X, Y, covariance_matrix, beta, predict_sample_size, alpha, lengthscale, kernel_variance):
    num_data = X.shape[0]
    mean = np.zeros(predict_sample_size)
    variance = np.zeros(predict_sample_size)
    x_sample_min = np.min(X) - abs(np.min(X)) * 0.15
    x_sample_max = np.max(X) + abs(np.max(X)) * 0.15
    x_sample = np.linspace(x_sample_min, x_sample_max, predict_sample_size)

    for sample in range(predict_sample_size):
        kernel = np.zeros((num_data, 1))
        for i in range(num_data):
            kernel[i][0] = rational_qudartic_kernel(X[i], x_sample[sample], alpha, lengthscale, kernel_variance)
        mean[sample] = mul(mul(kernel.T, inv(covariance_matrix)), Y)
        kernel_star = rational_qudartic_kernel(x_sample[sample], x_sample[sample], alpha, lengthscale, kernel_variance) + 1 / beta
        variance[sample] = kernel_star - mul(mul(kernel.T, inv(covariance_matrix)), kernel)

    return mean, variance

def GP_plot(X, Y, mean, variance, predict_sample_size):
    x_sample_min = np.min(X) - abs(np.min(X)) * 0.15
    x_sample_max = np.max(X) + abs(np.max(X)) * 0.15
    x_sample = np.linspace(x_sample_min, x_sample_max, predict_sample_size)
    interval = 1.96 * (variance ** 0.5)

    plt.scatter(X, Y, color = 'k')
    plt.plot(x_sample, mean, color = 'b')

    plt.plot(x_sample, mean + interval, color = 'r')
    plt.plot(x_sample, mean - interval, color = 'r')
    plt.fill_between(x_sample, mean + interval, mean - interval, color = 'pink', alpha = 0.3)
    
    plt.show()

    return

BETA = 5
ALPHA = 1
LENGTHSCALE = 1
KERNEL_VARIANCE = 1
PREDICT_SAMPLE_SIZE = 1000

X, Y = loadData("./data/input.data")
covariance_matrix = create_covariance_matrix(X, BETA, ALPHA, LENGTHSCALE, KERNEL_VARIANCE)
mean , variance = GP_predict(X, Y, covariance_matrix, BETA, PREDICT_SAMPLE_SIZE, ALPHA, LENGTHSCALE, KERNEL_VARIANCE)
GP_plot(X, Y, mean, variance, PREDICT_SAMPLE_SIZE)