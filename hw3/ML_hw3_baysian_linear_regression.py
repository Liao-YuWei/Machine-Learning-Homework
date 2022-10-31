import numpy as np

#Generate random data point using Irwin–Hall distribution
def random_error_generator(s):
    n = s * 12
    original_mean = n / 2
    original_random = sum(np.random.uniform(0, 1, n))   #A random data point ~N(n/2, s) where n = s * 12
    return original_random - original_mean

def random_data_generator(basis_num, error_variance, w):
    error = random_error_generator(error_variance)
    x = np.random.uniform(-0.99999999, 1)
    y = 0
    for i in range(basis_num):
        y += w[i] * (x ** i)
    y += error

    return x, y

print('For polynomial basis linear model, please enter following parameters')
basis_num = int(input('The number of basis: '))
error_variance = int(input('The error variance: '))
w = input('The coefficients: ')

w = w.split()
for i in range(len(w)):
    w[i] = int(w[i])
if basis_num!=len(w):
    print('Error: The number of basis and coefficients are not matched!')
    exit()

precision = int(input('\nEnter the precision for initial prior: '))

new_x, new_y = random_data_generator(basis_num, error_variance, w)
print(f'\nAdd data point ({new_x}, {new_y}):')

a = 1 / error_variance
design_matrix = []
Y = []
posterior_variance = np.zeros((basis_num, basis_num))
for i in range(basis_num):
    posterior_variance[i][i] = 1 / precision

