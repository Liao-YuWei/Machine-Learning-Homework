import numpy as np

print('For polynomial basis linear model data generator')
basis_num = int(input('Enter the basis: '))
a = int(input('Enter the error variance: '))
w = input('Enter the coefficients: ')

precision = int(input('Enter the precision for initial prior: '))

w = w.split()
for i in range(len(w)):
    w[i] = int(w[i])

