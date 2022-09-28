import numpy as np
import matplotlib as plt
import sys

def generateAb(file_path, n):
    data = open(file_path, 'r')
    datapoints = []
    for line in data:
        tmp = line.split(',')
        tmp = np.asarray(tmp, float)
        datapoints.append(tmp)
    
    A = np.ones((len(datapoints), n))
    for row in range(len(A)):
        for col in range(n-1):
            A[row][col] = datapoints[row][0] ** (n - col - 1)
    b = np.zeros((len(datapoints), 1))
    for row in range(len(b)):
        b[row][0] = datapoints[row][1]
    
    return A, b

def transpose(A):
    num_row, num_col = A.shape
    AT = np.zeros((num_col, num_row))
    for row in range(num_col):
        for col in range(num_row):
            AT[row][col] = A[col][row]
    return AT

def matrixMulti(A, B):
    try:
        num_row, num_col = A.shape[0], B.shape[1]
        result = np.zeros((num_col, num_row))
        for row in range(num_row):
            for col in range(num_col):
                for i in range(A.shape[1]):
                    result[row][col] += A[row][i] * B[i][col]
    except IndexError:
        print('IndexError at matrixMulti, the size of 2 matrices may not match')
    return result


def matrixAdd(A, B):
    try:
        num_row, num_col = A.shape
        result = np.zeros((num_col, num_row))
        for row in range(num_row):
            for col in range(num_col):
                result[row][col] = A[row][col] + B[row][col]
    except IndexError:
        print('IndexError at matrixAdd, the size of 2 matrices may not match')
    return result


file_path = input('Please enter file path and names: ')
n = int(input('Please enter the number of polynomial bases: '))
lambda_ = float(input('Please enter lambda: '))

A, b = generateAb(file_path, n)

AT = transpose(A)

"""
LSE
"""
lambdaI = lambda_ * np.identity(A.shape[1])
x = matrixAdd(matrixMulti(AT, A), lambdaI)



