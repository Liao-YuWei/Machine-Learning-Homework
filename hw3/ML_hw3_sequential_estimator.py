import numpy as np

def random_data_generator(m, s):
    n = s * 12
    original_mean = n / 2
    offset = m - original_mean
    return sum(np.random.uniform(0, 1, n)) + offset 

mean = int(input('Enter the mean of gaussian for univariate gaussian data generator: '))
variance = int(input('Enter the variance of gaussian for univariate gaussian data generator: '))

