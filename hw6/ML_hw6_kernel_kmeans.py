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

def construct_C(clusters):
    C = np.zeros(NUM_CLUSTER, dtype=int)
    for i in range(IMAGE_SIZE):
        C[clusters[i]] += 1

    return C

def init_alpha_c(centers):
    # alpha = np.zeros((IMAGE_SIZE, NUM_CLUSTER), dtype=int)
    # for i in range(NUM_CLUSTER):
    #     alpha[centers[i]][i] = 1
    # clusters = np.sum(alpha, axis=0, dtype=int) #sum up each of column

    clusters = np.zeros(IMAGE_SIZE, dtype=int)
    for i in range(NUM_CLUSTER):
        clusters[centers[i]] = i
    C = construct_C(clusters)

    return clusters, C

def init(img_kernel):
    if MODE == 'random':
        centers = np.random.randint(IMAGE_SIZE, size=NUM_CLUSTER)
        clusters, C = init_alpha_c(centers)
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

        # alpha, clusters = init_alpha_c(centers)
        clusters, C = init_alpha_c(centers)
    else:
        print('Wrong input for cluster initialization!')

    # return alpha, clusters
    return clusters, C

def sigma_n(img_kernel, clusters, j, k):
    sum = 0
    for n in range(IMAGE_SIZE):
        if clusters[n] == k:
            sum += img_kernel[j][n]
    
    return sum

def sigma_pq(img_kernel, clusters, k):
    sum = 0
    group = np.where(clusters == k)
    print(group)
    progress = tqdm(total=np.shape(group)[0] * np.shape(group)[0])
    for p in np.nditer(group):
        for q in np.nditer(group):
            sum += img_kernel[p][q]
            progress.update(1)
    # for p in range(IMAGE_SIZE):
    #     for q in range(IMAGE_SIZE):
    #         if alpha[p][k] == 1 and alpha[q][k] == 1:
    #             sum += img_kernel[p][q]
    print('end pq')

    return sum

def kernel_kmeans(img_kernel, clusters, C):
    # new_alpha = np.zeros((IMAGE_SIZE, NUM_CLUSTER), dtype=int)
    new_clusters = np.zeros(IMAGE_SIZE, dtype=int)
    for pixel in range(IMAGE_SIZE):
        distances = np.zeros(NUM_CLUSTER)
        for cluster in trange(NUM_CLUSTER):
            distances[cluster] = img_kernel[pixel][pixel]
            distances[cluster] += 2 / C[cluster] * sigma_n(img_kernel, clusters, pixel, cluster)
            distances[cluster] += 1 / (C[cluster] ** 2) * sigma_pq(img_kernel, clusters, cluster)
        new_clusters[pixel] = np.argmin(distances)
        # cur_cluster = np.argmin(distances)
        # new_alpha[pixel][cur_cluster] = 1
    # new_clusters = np.sum(new_alpha, axis=0, dtype=int)
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
# alpha, clusters = init(img_kernel)
clusters, C = init(img_kernel)
# print(clusters)
# print(C)

while not converge:
    print(f'iteration: {iteration}')
    pre_clusters = clusters

    # alpha, clusters = kernel_kmeans(img_kernel, alpha, clusters)

    # converge = check_converge(alpha, pre_alpha)
    # save_picture(alpha, iteration)
    clusters, C = kernel_kmeans(img_kernel, clusters, C)
    # print(clusters)
    # print(C)

    converge = check_converge(clusters, pre_clusters)
    save_picture(clusters, iteration)
    iteration += 1