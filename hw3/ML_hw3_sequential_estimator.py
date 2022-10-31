from operator import truth
import numpy as np

#Generate random data point using Irwin–Hall distribution
def random_data_generator(m, s):
    n = s * 12
    original_mean = n / 2
    offset = m - original_mean
    original_random = sum(np.random.uniform(0, 1, n))   #A random data point ~N(n/2, s) where n = s * 12
    return original_random + offset

print('For univariate gaussian data generator, please enter following parameters')
truth_mean = int(input('The mean: '))
truth_variance = int(input('The variance: '))

print(f'Data point source function: N({truth_mean}, {truth_variance})', end = '\n\n')

n = 0
predict_mean = 0
predict_variance = 0

while True:
    new_data = random_data_generator(truth_mean, truth_variance)
    print(f'Add data point: {new_data}')
    n += 1
    predict_mean = (predict_mean * (n-1) + new_data) / n
    sqr_error = (new_data - predict_mean) ** 2
    predict_variance = (predict_variance * (n-1) + sqr_error) / n
    print(f'Mean = {predict_mean}   Variance = {predict_variance}', end = '\n\n')

    if abs(truth_mean - predict_mean) < 5e-2 and abs(truth_variance - predict_variance) < 5e-2:
        break