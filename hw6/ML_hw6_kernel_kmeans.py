import math
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import os

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

def init_alpha_c(centers):
    alpha = np.zeros((IMAGE_SIZE, NUM_CLUSTER), dtype=int)
    for i in range(NUM_CLUSTER):
        alpha[centers[i]][i] = 1
    clusters = np.sum(alpha, axis=0, dtype=int) #sum up each of column

    return alpha, clusters

def init(img_kernel):
    if MODE == 'random':
        centers = np.random.randint(IMAGE_SIZE, size=NUM_CLUSTER)
        alpha, clusters = init_alpha_c(centers)
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

        alpha, clusters = init_alpha_c(centers)
    else:
        print('Wrong input for cluster initialization!')

    return alpha, clusters

def sigma_n(img_kernel, alpha, j, k):
    sum = 0
    for n in range(IMAGE_SIZE):
        if alpha[n][k] == 1:
            sum += img_kernel[j][n]
    
    return sum

def sigma_pq(img_kernel, alpha, k):
    sum = 0
    for p in range(IMAGE_SIZE):
        for q in range(IMAGE_SIZE):
            if alpha[p][k] == 1 and alpha[q][k] == 1:
                sum += img_kernel[p][q]

    return sum

def kernel_kmeans(img_kernel, alpha, clusters):
    new_alpha = np.zeros((IMAGE_SIZE, NUM_CLUSTER), dtype=int)
    for img in trange(IMAGE_SIZE):
        distances = np.zeros(NUM_CLUSTER)
        for cluster in range(NUM_CLUSTER):
            distances[cluster] = img_kernel[img][img]
            distances[cluster] += 2 / clusters[cluster] * sigma_n(img_kernel, alpha, img, cluster)
            distances[cluster] += 1 / (clusters[cluster] ** 2) * sigma_pq(img_kernel, alpha, cluster)
        cur_cluster = np.argmin(distances)
        new_alpha[img][cur_cluster] = 1
    new_clusters = np.sum(new_alpha, axis=0, dtype=int)
        
    return new_alpha, new_clusters

def check_converge(alpha, pre_alpha):
    for image in range(IMAGE_SIZE):
        if np.argmax(alpha[image]) != np.argmax(pre_alpha[image]):
            return 0
    
    return 1

def save_picture(alpha, iteration):
    pixel = np.zeros((10000, 3))
    for i in range(IMAGE_SIZE):
        cluster = np.argmax(alpha[i])
        pixel[i, :] = COLOR[cluster, :]
    pixel = np.reshape(pixel, (100, 100, 3))
    img = Image.fromarray(np.uint8(pixel))
    img.save(os.path.join(OUTPUT_DIR, '%06d.png'%iteration))

    return

IMAGE_SIZE = 10000
GAMMA_S = 1 / IMAGE_SIZE
GAMMA_C = 1 / (256 ** 3)
NUM_CLUSTER = 2
MODE = 'k-means++'
COLOR = [[56, 207, 0], [64, 70, 230], [186, 7, 61], [245, 179, 66], [55, 240, 240]]
OUTPUT_DIR = './output/kmeans'

converge = 0
iteration = 1

img = np.asarray(Image.open('.\data\image1.png').getdata()) #load image to 10000*3 numpy array

img_kernel = kernel(img)
alpha, clusters = init(img_kernel)

while not converge:
    print(f'iteration: {iteration}')
    pre_alpha = alpha
    pre_clusters = clusters

    alpha, clusters = kernel_kmeans(img_kernel, alpha, clusters)

    converge = check_converge(alpha, pre_alpha)
    save_picture(alpha, iteration)
    iteration += 1