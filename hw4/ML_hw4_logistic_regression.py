from os import cpu_count
import numpy as np
from numpy.linalg import inv
from numpy import matmul as mul
import math

#Generate random data point using Irwin–Hall distribution
def random_error_generator(mean, var):
    n = int(var * 12)
    original_mean = n / 2
    offset = mean - original_mean
    original_random = sum(np.random.uniform(0, 1, n))   #A random data point ~N(n/2, s) where n = s * 12

    return original_random + offset

#Create design matrix and label matrix
def create_design_matrix_y(D1, D2):
    num_data = D1.shape[0]
    design_matrix = np.zeros((num_data * 2, 3))
    y = np.zeros((num_data * 2, 1))
    for i in range(num_data):
        design_matrix[i][0] = 1
        design_matrix[i][1] = D1[i][0]
        design_matrix[i][2] = D1[i][1]
        design_matrix[i + num_data][0] = 1
        design_matrix[i + num_data][1] = D2[i][0]
        design_matrix[i + num_data][2] = D2[i][1]
        y[i + num_data] = 1

    return design_matrix, y

def transpose(A):
    num_row, num_col = A.shape
    AT = np.zeros((num_col, num_row))
    for row in range(num_col):
        for col in range(num_row):
            AT[row][col] = A[col][row]
    return AT

def gradient(design_matrix, y, weight, n):
    wx= mul(design_matrix, weight)
    sigmoid = np.empty((n * 2, 1))
    for i in range(n * 2):
        sigmoid[i] = 1 / (1 + math.exp(-1 * wx[i]))
    At = transpose(design_matrix)
    grad = mul(At, (sigmoid - y))
    return grad

def converge(weight, weight_pre):
    for (i, j) in zip(weight, weight_pre):
        if abs(i - j) > (abs(j) * 0.07):
            return False
    
    return True

def gradient_descent(design_matrix, y, n):
    weight_pre = np.zeros((3, 1))
    weight = np.zeros((3, 1))
    learning_rate = 0.05

    count = 0
    while True:
        count += 1
        weight -= learning_rate * gradient(design_matrix, y, weight, n)
        if converge(weight, weight_pre):
            break
        weight_pre = np.copy(weight)
    print(count)
    
    return weight

def test(weight, design_matrix, n):
    prediction = np.empty((n * 2, 1), dtype = int)
    wt = transpose(weight)

    for i in range(n * 2):
        x = mul(wt, design_matrix[i])
        activate = 1 / (1 + math.exp(-1 * x))
        cur_prediction = 1 if activate > 0.5 else 0
        prediction[i] = cur_prediction
    
    return prediction

def confusion_matrix(y, prediction, n):
    confusion = np.zeros((2, 2))
    for i in range(n * 2):
        if y[i] == 0:
            if y[i] == prediction[i]:   #TP
                confusion[0][0] += 1
            else:                       #FN
                confusion[0][1] += 1
        else:
            if y[i] == prediction[i]:   #TN
                confusion[1][1] += 1
            else:                       #FP
                confusion[1][0] += 1
    
    return confusion

def print_result(weight, confusion):
    print('w:')
    for i in range(weight.shape[0]):
        print(weight[i])

    print('\nConfusion Matrix:')
    print('\t\tPredict cluster 1\tPredict cluster 2')
    print(f'In cluster 1\t\t{confusion[0][0]}\t\t\t{confusion[0][1]}')
    print(f'In cluster 2\t\t{confusion[1][0]}\t\t\t{confusion[1][1]}')

    sensitivity = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    specificity = confusion[1][1] / (confusion[1][0] + confusion[1][1])
    print(f'\nSensitivity (Successfully predict cluster 1): {sensitivity}')
    print(f'Specificity (Successfully predict cluster 2): {specificity}')

n = int(input('Number of data points: '))
mx1 = float(input('mx1: ' ))
vx1 = float(input('vx1: '))
my1 = float(input('my1: '))
vy1 = float(input('vy1: '))
mx2 = float(input('mx2: '))
vx2 = float(input('vx2: '))
my2 = float(input('my2: '))
vy2 = float(input('vy2: '))

D1 = np.empty((0, 2))
D2 = np.empty((0, 2))
for i in range(n):
    cur_x1 = random_error_generator(mx1, vx1)
    cur_y1 = random_error_generator(my1, vy1)
    cur_x2 = random_error_generator(mx2, vx2)
    cur_y2 = random_error_generator(my2, vy2)
    D1 = np.vstack([D1, [cur_x1, cur_y1]])
    D2 = np.vstack([D2, [cur_x2, cur_y2]])

design_matrix, y = create_design_matrix_y(D1, D2)

gradient_w = gradient_descent(design_matrix, y, n)
gradient_prediction = test(gradient_w, design_matrix, n)
gradient_confusion = confusion_matrix(y, gradient_prediction, n)

print('Gradient descent:')
print_result(gradient_w, gradient_confusion)