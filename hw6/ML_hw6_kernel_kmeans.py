import math
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import os
import glob
from scipy.spatial.distance import cdist

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
    color_dist = cdist(img, img, 'sqeuclidean')
    coordinates = np.zeros((IMAGE_SIZE, 2))
    for row in range(100):
        row_index = row * 100
        for col in range(100):
            coordinates[row_index + col][0] = row
            coordinates[row_index + col][1] = col
    spatial_distance = cdist(coordinates, coordinates, 'sqeuclidean')
    img_kernel = np.multiply(np.exp(-GAMMA_C * color_dist), np.exp(-GAMMA_S * spatial_distance))

    return img_kernel

def construct_C(clusters):
    C = np.zeros(NUM_CLUSTER, dtype=int)
    for i in range(IMAGE_SIZE):
        C[clusters[i]] += 1

    return C

def init_cluster_c(img_kernel, centers):
    clusters = np.zeros(IMAGE_SIZE, dtype=int)
    for pixel in range(IMAGE_SIZE):
        if pixel in centers:
            continue
        min_dist = np.Inf
        for cluster in range(NUM_CLUSTER):
            center = centers[cluster]
            temp_dist = img_kernel[pixel][center]
            if temp_dist < min_dist:
                min_dist = temp_dist
                clusters[pixel] = cluster

    C = construct_C(clusters)

    save_picture(clusters, 0)

    return clusters, C

def init(img_kernel):
    if MODE == 'random':
        centers = np.random.randint(IMAGE_SIZE, size=NUM_CLUSTER)
        clusters, C = init_cluster_c(img_kernel, centers)
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
        clusters, C = init_cluster_c(img_kernel, centers)
    else:
        print('Wrong input for cluster initialization!')

    return clusters, C

def sigma_n(pixel_kernel, clusters, k, C):
    sum = 0
    for n in range(IMAGE_SIZE):
        if clusters[n] == k:
            sum += pixel_kernel[n]
    
    return 2 / C[k] * sum

def sigma_pq(img_kernel, clusters, C):
    sum = np.zeros(NUM_CLUSTER)
    for k in range(NUM_CLUSTER):
        ker = img_kernel.copy()
        for pixel in range(IMAGE_SIZE):
            if clusters[pixel] != k:
                ker[pixel, :] = 0
                ker[:, pixel] = 0
        sum[k] = np.sum(ker)/ (C[k] **2)

    return sum

def kernel_kmeans(img_kernel, clusters, C):
    new_clusters = np.zeros(IMAGE_SIZE, dtype=int)
    pq = sigma_pq(img_kernel, clusters, C)
    for pixel in trange(IMAGE_SIZE):
        distances = np.zeros(NUM_CLUSTER)
        for cluster in range(NUM_CLUSTER):
            distances[cluster] = img_kernel[pixel][pixel]
            distances[cluster] -= sigma_n(img_kernel[pixel, :], clusters, cluster, C)
            distances[cluster] += pq[cluster]
        new_clusters[pixel] = np.argmin(distances)
    new_C = construct_C(new_clusters)
        
    return new_clusters, new_C

def check_converge(clusters, pre_clusters):
    for pixel in range(IMAGE_SIZE):
        if clusters[pixel] != pre_clusters[pixel]:
            return 0
    
    return 1

def save_picture(clusters, iteration):
    pixel = np.zeros((10000, 3))
    for i in range(IMAGE_SIZE):
        pixel[i, :] = COLOR[clusters[i], :]
    pixel = np.reshape(pixel, (100, 100, 3))
    img = Image.fromarray(np.uint8(pixel))
    img.save(OUTPUT_DIR + '\%01d_%03d.png'%(NUM_CLUSTER, iteration), 'png')

    return

IMAGE_SIZE = 10000
COLOR = np.array([[56, 207, 0], [64, 70, 230], [186, 7, 61], [245, 179, 66], [55, 240, 240]])

IMAGE_ID = 1
GAMMA_S = 0.001
GAMMA_C = 0.001
NUM_CLUSTER = 4
MODE = 'random'
IMAGE_PATH = f'.\data\image{IMAGE_ID}.png'
OUTPUT_DIR = f'.\output\kmeans\{MODE}\image{IMAGE_ID}'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

converge = 0
iteration = 1

print(f'Image: {IMAGE_ID}, k: {NUM_CLUSTER}, mode: {MODE}')
img = np.asarray(Image.open(IMAGE_PATH).getdata()) #load image to 10000*3 numpy array

img_kernel = kernel(img)
clusters, C = init(img_kernel)

while not converge:
    print(f'iteration: {iteration}')
    pre_clusters = clusters

    clusters, C = kernel_kmeans(img_kernel, clusters, C)

    save_picture(clusters, iteration)
    converge = check_converge(clusters, pre_clusters)
    
    iteration += 1