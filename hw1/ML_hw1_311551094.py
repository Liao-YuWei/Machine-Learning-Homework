import numpy as np
import matplotlib.pyplot as plt

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
    
    datapoints = np.array(datapoints)
    
    return A, b, datapoints

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
        result = np.zeros((num_row, num_col))
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
        result = np.zeros((num_row, num_col))
        for row in range(num_row):
            for col in range(num_col):
                result[row][col] = A[row][col] + B[row][col]
    except IndexError:
        print('IndexError at matrixAdd, the size of 2 matrices may not match')
    return result

def LUInverse(A):
    temp_A = A.copy()
    matrix_size = temp_A.shape[0]
    # We can skip L and find L_inv directly.
    L_inv = np.identity(matrix_size)
    for row in range(1, matrix_size):
        for col in range(row):
            E = np.identity(matrix_size)
            E[row][col] = -(temp_A[row][col] / temp_A[col][col])
            temp_A = matrixMulti(E, temp_A)
            L_inv[row][col] = E[row][col]

    #Now, temp_A is U. Then, calculate U_inv from U.
    U = temp_A
    U_inv = np.identity(matrix_size)
    for col in range(matrix_size):
        U_inv[col][col] = 1/U[col][col]
        for row in range(col-1, -1, -1):
            sum = 0
            for i in range(row+1, col+1):
                sum += U[row][i] * U_inv[i][col]
            U_inv[row][col] = -(sum/U[row][row])
    
    result = matrixMulti(U_inv, L_inv)
    
    return result

def calculateError(A, X, b):
    error_matrix = matrixAdd(matrixMulti(A, X), -b)
    error = matrixMulti(transpose(error_matrix), error_matrix)
    return error[0][0]

def printResult(X, error):
    print('Fitting Line: ', end = '')
    for i in range(X.shape[0]):
        print('%.3f' % X[i], end = '')
        if i != X.shape[0] - 1:
            print(' X^%d + ' % (X.shape[0] - 1 - i), end = '')
    print('\nTotal Error: %.3f' % error)

def drawPlot(fig, X, datapoints, num, method):
    thisPlt = fig.add_subplot(2, 1, num)
    thisPlt.scatter(datapoints[:, 0], datapoints[:, 1], c = 'r')
    x_min = np.min(datapoints, axis=0)[0]
    x_max = np.max(datapoints, axis=0)[0]
    x_range = x_max - x_min
    x_line = np.linspace(x_min - x_range * 0.1, x_max + x_range * 0.1, 50000)
    y_line = np.zeros(len(x_line))

    for i in range(len(x_line)):
        for j in range(X.shape[0]):
            y_line[i] += X[j] * (x_line[i] ** (X.shape[0] - 1 - j))
    thisPlt.plot(x_line, y_line, c = 'k')

    thisPlt.set_xlim(x_min - x_range * 0.1, x_max + x_range * 0.1)
    thisPlt.set_title(method)

"""
Program starts here
"""
file_path = input('Please enter file path and names: ')
n = int(input('Please enter the number of polynomial bases: '))
lambda_ = float(input('Please enter lambda: '))

A, b, datapoints = generateAb(file_path, n)
AT = transpose(A)

fig = plt.figure()

"""
LSE
"""
lambdaI = lambda_ * np.identity(A.shape[1])
ATA_lamdaI = matrixAdd(matrixMulti(AT, A), lambdaI)
ATA_lamdaI_inv = LUInverse(ATA_lamdaI)
ATb = matrixMulti(AT, b)
X_LSE = matrixMulti(ATA_lamdaI_inv, ATb)

error_LSE = calculateError(A, X_LSE, b)

print('\nLSE:')
printResult(X_LSE, error_LSE)
drawPlot(fig, X_LSE, datapoints, 1, "LSE")

"""
Newton's Method
"""
ATA = matrixMulti(AT, A)
ATA_inv = LUInverse(ATA)
ATb = matrixMulti(AT, b)
X_Newton = matrixMulti(ATA_inv, ATb)

error_Newton = calculateError(A, X_Newton, b)

print("\nNewton's Method:")
printResult(X_Newton, error_Newton)
drawPlot(fig, X_Newton, datapoints, 2, "Newton's method")

plt.subplots_adjust(hspace = 0.33)
plt.show()