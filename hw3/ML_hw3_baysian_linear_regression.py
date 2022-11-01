import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import matmul as mul

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

def transpose(A):
    num_row, num_col = A.shape
    AT = np.zeros((num_col, num_row))
    for row in range(num_col):
        for col in range(num_row):
            AT[row][col] = A[col][row]
    return AT

def matrix_add(A, B):
    try:
        num_row, num_col = A.shape
        result = np.zeros((num_row, num_col))
        for row in range(num_row):
            for col in range(num_col):
                result[row][col] = A[row][col] + B[row][col]
    except IndexError:
        print('IndexError at matrix_add, the size of 2 matrices may not match')
    return result

def print_posterior(posterior_mean, posterior_variance):
    print('Posterior mean:')
    for mean in posterior_mean:
        print(float(mean))

    print('\nPosterior variance:')
    for row in posterior_variance:
        for col in row:
            print(col, end='  ')
        print()
    
    return

def ground_truth_plot(coefficient, error_variance, basis_num):
    plt.subplot(221)
    plt.title("Ground Truth")
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    
    x_curve = np.linspace(-2.0, 2.0, 30)
    y_curve = np.zeros(30)
    y_up_var = np.zeros(30)
    y_down_var = np.zeros(30)

    for i in range(30):
        for n in range(basis_num):
            y_curve[i] += coefficient[n] * (x_curve[i] ** n)
        y_up_var[i] = y_curve[i] + error_variance
        y_down_var[i] = y_curve[i] - error_variance

    plt.plot(x_curve, y_curve, 'k')
    plt.plot(x_curve, y_up_var, 'r')
    plt.plot(x_curve, y_down_var, 'r')

    return

def predict_plot(position, title, x_points, y_points, coefficient_mean, coefficient_variance, basis_num):
    plt.subplot(position)
    plt.title(title)
    plt.xlim(-2.0, 2.0)
    plt.ylim(-20.0, 20.0)
    
    x_curve = np.linspace(-2.0, 2.0, 30)
    y_curve = np.zeros(30)
    y_up_var = np.zeros(30)
    y_down_var = np.zeros(30)

    for i in range(30):
        tmp_design_matrix = np.array([x_curve[i] ** j for j in range(basis_num)])
        A_sigma_At = float(mul(tmp_design_matrix, mul(coefficient_variance, tmp_design_matrix.T)))
        distance = 1 / a + A_sigma_At
        y_curve[i] = mul(tmp_design_matrix, coefficient_mean)
        y_up_var[i] = y_curve[i] + distance
        y_down_var[i] = y_curve[i] - distance

    plt.plot(x_curve, y_curve, 'k')
    plt.plot(x_curve, y_up_var, 'r')
    plt.plot(x_curve, y_down_var, 'r')

    plt.scatter(x_points, y_points, color = 'b')

    return


print('For polynomial basis linear model, please enter following parameters')
basis_num = int(input('The number of basis:'))
error_variance = int(input('The error variance:'))
w = input('The coefficients:')

w = w.split()
for i in range(len(w)):
    w[i] = int(w[i])
if basis_num!=len(w):
    print('Error: The number of basis and coefficients are not matched!')
    exit()

precision = int(input('\nEnter the precision for initial prior:'))

a = 1 / error_variance
design_matrix = np.zeros((1, basis_num))

#store all random data points so for to variable x and y
x = []
y = []

posterior_variance = np.zeros((basis_num, basis_num))
for i in range(basis_num):
    posterior_variance[i][i] = 1 / precision
posterior_mean = np.zeros((basis_num, 1))

pre_predictive_variance = 0

count = 0

while True:
    #generate new data
    new_x, new_y = random_data_generator(basis_num, error_variance, w)
    x = np.append(x, new_x)
    y = np.append(y, new_y)
    print(f'\nAdd data point ({new_x}, {new_y}):')

    #add x of new data to design matrix
    for i in range(basis_num):
        design_matrix[0][i] = new_x ** i
    
    #calculate posterior variance and mean
    prior_variance = posterior_variance
    At = design_matrix.T
    a_At_A = a * mul(At, design_matrix)
    S = inv(prior_variance)
    posterior_variance = inv(matrix_add(a_At_A, S))

    prior_mean = posterior_mean
    a_At_y = a * mul(At, [[new_y]])
    S_m = mul(S, prior_mean)
    posterior_mean = mul(posterior_variance, matrix_add(a_At_y, S_m))

    print_posterior(posterior_mean, posterior_variance)

    #calculate mean and variance of predictive distribution
    predictive_mean = float(mul(design_matrix, posterior_mean))
    A_sigma_At = float(mul(design_matrix, mul(posterior_variance, At)))
    predictive_variance = 1 / a + A_sigma_At

    print(f'\nPredictive distribution ~ N({predictive_mean}, {predictive_variance})')

    #break if predictive_variance change is small(already converge)
    if abs(predictive_variance - pre_predictive_variance) < 1e-5 and count > 50:
        break

    pre_predictive_variance = predictive_variance
    count += 1

    if count == 10:
        x_10 = np.copy(x)
        y_10 = np.copy(y)
        posterior_mean_10 = np.copy(posterior_mean)
        posterior_variance_10 = np.copy(posterior_variance)
    elif count == 50:
        x_50 = np.copy(x)
        y_50 = np.copy(y)
        posterior_mean_50 = np.copy(posterior_mean)
        posterior_variance_50 = np.copy(posterior_variance)

#plot result graph
ground_truth_plot(w, error_variance, basis_num)
predict_plot(222, 'Final Predict Result', x, y, posterior_mean, posterior_variance, basis_num)
predict_plot(223, 'After 10 incomes', x_10, y_10, posterior_mean_10, posterior_variance_10, basis_num)
predict_plot(224, 'After 50 incomes', x_50, y_50, posterior_mean_50, posterior_variance_50, basis_num)

plt.show()