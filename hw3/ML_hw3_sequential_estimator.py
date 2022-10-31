from operator import truth
import numpy as np

def random_data_generator(m, s):
    n = s * 12
    original_mean = n / 2
    offset = m - original_mean
    return sum(np.random.uniform(0, 1, n)) + offset 

truth_mean = int(input('Enter the mean of gaussian for univariate gaussian data generator: '))
truth_variance = int(input('Enter the variance of gaussian for univariate gaussian data generator: '))

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