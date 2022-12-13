import math
import numpy as np
from PIL import Image
from tqdm import tqdm

def spatial_rbf(gamma, x1, x2):
    x1_row = x1 // 100
    x1_col = x1 % 100
    x2_row = x2 // 100
    x2_col = x2 % 100
    square = (x1_row - x2_row) ** 2 + (x1_col - x2_col) ** 2

    return math.exp(-gamma * square)

def color_rbf(gamma, x1, x2):
    square = 0
    for i in range(3):
        square += (x1[i] - x2[i]) ** 2
    
    return math.exp(-gamma * square)

def kernel(img):
    img_kernel = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    progress = tqdm(total=50005000)
    
    for row in range(IMAGE_SIZE):
        for col in range(row, IMAGE_SIZE):
            rbf_spatial = spatial_rbf(GAMMA_S, row, col)
            rbf_color = color_rbf(GAMMA_C, img[row], img[col])
            img_kernel[row][col] = rbf_spatial * rbf_color
            img_kernel[col][row] = img_kernel[row][col]
            progress.update(1)
    
    return img_kernel

def init_center(img_kernel):
    if MODE == 'random':
        centers = np.random.randint(IMAGE_SIZE, size=NUM_CLUSTER)
    elif MODE == 'k-means++':
        centers = np.zeros(NUM_CLUSTER, dtype=int)
        centers[0] = np.random.randint(IMAGE_SIZE, size=1)
        for i in range(1, NUM_CLUSTER):
            distances = np.zeros(IMAGE_SIZE)
            for j in range(IMAGE_SIZE):
                min_dist = np.Inf
                for k in range(i):
                    temp_dist = img_kernel[centers[k]][j]
                    if temp_dist < min_dist:
                        min_dist = temp_dist
                distances[j] = min_dist
            distances = distances / np.sum(distances)
            centers[i] = np.random.choice(10000, size=1, p=distances)
    else:
        print('Wrong input for cluster initialization!')

    return centers

IMAGE_SIZE = 10000
GAMMA_S = 1 / IMAGE_SIZE
GAMMA_C = 1 / (256 ** 3)
NUM_CLUSTER = 5
MODE = 'k-means++'

img = np.asarray(Image.open('.\data\image1.png').getdata()) #load image to 10000*3 numpy array

img_kernel = kernel(img)
init_mean = init_center(img_kernel)
print(init_mean)