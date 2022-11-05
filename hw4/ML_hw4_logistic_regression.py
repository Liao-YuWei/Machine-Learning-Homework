import numpy as np
from numpy.linalg import inv
from numpy import matmul as mul
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    labels = np.zeros((num_data * 2, 1))
    for i in range(num_data):
        design_matrix[i][0] = 1
        design_matrix[i][1] = D1[i][0]
        design_matrix[i][2] = D1[i][1]
        design_matrix[i + num_data][0] = 1
        design_matrix[i + num_data][1] = D2[i][0]
        design_matrix[i + num_data][2] = D2[i][1]
        labels[i + num_data] = 1

    return design_matrix, labels

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
    #print(count)
    
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

def hessian_matrix(design_matrix, weight, n):
    D = np.zeros((n, n))
    for i in range(n):
        wx= mul(design_matrix[i], weight)
        D[i][i] = math.exp(-wx) / ((1 + math.exp(-wx)) ** 2)

    At = transpose(design_matrix)
    hessian = mul(mul(At, D), design_matrix)

    return hessian

def determinant(matrix):
    result = 0
    result += matrix[0][0] * matrix[1][1] * matrix[2][2]
    result += matrix[0][1] * matrix[1][2] * matrix[2][0]
    result += matrix[0][2] * matrix[1][0] * matrix[2][1]
    result -= matrix[0][2] * matrix[1][1] * matrix[2][0]
    result -= matrix[0][0] * matrix[1][2] * matrix[2][1]
    result -= matrix[0][1] * matrix[1][0] * matrix[2][2]

    return result

def newtons_method(design_matrix, y, n):
    weight_pre = np.zeros((3, 1))
    weight = np.zeros((3, 1))
    learning_rate = 0.05

    count = 0
    while True:
        count += 1
        grad = gradient(design_matrix, y, weight, n)
        hessain = hessian_matrix(design_matrix, weight, n * 2)
        
        if determinant(hessain) != 0:
            weight -= learning_rate * mul(inv(hessain), grad)
        else:
            weight -= learning_rate * grad

        if converge(weight, weight_pre):
            break
        weight_pre = np.copy(weight)
    #print(count)

    return weight

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

design_matrix, labels = create_design_matrix_y(D1, D2)

#gradient descent
gradient_w = gradient_descent(design_matrix, labels, n)
gradient_prediction = test(gradient_w, design_matrix, n)
gradient_confusion = confusion_matrix(labels, gradient_prediction, n)

print('Gradient descent:')
print_result(gradient_w, gradient_confusion)

#newton's method
newtons_w = newtons_method(design_matrix, labels, n)
newtons_prediction = test(newtons_w, design_matrix, n)
newtons_confusion = confusion_matrix(labels, newtons_prediction, n)

print('------------------------')
print("Newton's method:")
print_result(newtons_w, newtons_confusion)

whole_data = np.vstack((D1, D2))
color = ['red','blue']

cur_subplot = plt.subplot(131)
cur_subplot.scatter(whole_data[:, 0], whole_data[:, 1], c=labels, cmap=colors.ListedColormap(color))
cur_subplot.set_title('Ground Truth')

cur_subplot = plt.subplot(132)
cur_subplot.scatter(whole_data[:, 0], whole_data[:, 1], c=gradient_prediction, cmap=colors.ListedColormap(color))
cur_subplot.set_title('Gradient descent')

cur_subplot = plt.subplot(133)
cur_subplot.scatter(whole_data[:, 0], whole_data[:, 1], c=newtons_prediction, cmap=colors.ListedColormap(color))
cur_subplot.set_title("Newton's Method")

plt.show()