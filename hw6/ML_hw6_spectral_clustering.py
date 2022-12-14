import numpy as np
from PIL import Image
import os
from scipy.spatial.distance import cdist

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

def compute_laplacian(W):
    L = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    D = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for i in range(IMAGE_SIZE):
        D[i][i] = np.sum(W[i, :])
    L = D - W

    return L, D

def normalize_laplacian(L, D):
    sqrt_D = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for i in range(100):
        sqrt_D[i][i] = D[i][i] ** (-0.5)
    L_n = sqrt_D @ L @ sqrt_D

    return L_n

def eigen_decomposition(L):
    eigenvalues, eigenvectors = np.linalg.eig(L)
    index = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, index]

    return eigenvectors[:, 1:1+NUM_CLUSTER].real

def init_means_clusters(U, centers):
    means = np.zeros((NUM_CLUSTER, NUM_CLUSTER))
    clusters = np.full(IMAGE_SIZE, -1, dtype=int)

    for i in range(NUM_CLUSTER):
        means[i] = U[centers[i]]
        clusters[centers[i]] = i

    return means, clusters

def square_distance(center, cur_point):
    distance = 0
    for i in range(NUM_CLUSTER):
        distance += (center[i] - cur_point[i]) ** 2
    
    return distance

def init_kmeans(U):
    if MODE == 'random':
        centers = np.random.randint(IMAGE_SIZE, size=NUM_CLUSTER)
        means, clusters = init_means_clusters(U, centers)
    elif MODE == 'k-means++':
        centers = np.zeros(NUM_CLUSTER, dtype=int)
        centers[0] = np.random.randint(IMAGE_SIZE, size=1)

        for i in range(1, NUM_CLUSTER):
            distances = np.zeros(IMAGE_SIZE)
            for j in range(IMAGE_SIZE):
                min_dist = np.Inf
                for k in range(i):
                    temp_dist = square_distance(U[centers[k]], U[j])
                    if temp_dist < min_dist:
                        min_dist = temp_dist
                distances[j] = min_dist
            distances = distances / np.sum(distances)
            centers[i] = np.random.choice(10000, size=1, p=distances)
        means, clusters = init_means_clusters(U, centers)
    else:
        print('Wrong input for cluster initialization!')

    # return means, clusters
    return centers, clusters

def E_step(U, means):
    new_clusters = np.zeros(IMAGE_SIZE, dtype=int)
    for pixel in range(IMAGE_SIZE):
        min_dist = np.Inf
        for cluster in range(NUM_CLUSTER):
            temp_dist = square_distance(means[cluster], U[pixel])
            if temp_dist < min_dist:
                min_dist = temp_dist
                new_clusters[pixel] = cluster

    # save_picture(clusters, 0)

    return new_clusters

def M_step(U, clusters):
    new_means = np.zeros((NUM_CLUSTER, NUM_CLUSTER))
    for cluster in range(NUM_CLUSTER):
        count = 0
        for pixel in range(IMAGE_SIZE):
            if clusters[pixel] == cluster:
                new_means[cluster] += U[pixel]
                count += 1
        new_means[cluster] /= count

    return new_means

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

def kmeans(U):
    converge = 0
    iteration = 1
    means, clusters = init_kmeans(U)

    # while not converge:
    #     print(f'iteration: {iteration}')
    #     pre_clusters = clusters

    #     clusters = E_step(U, means)
    #     means = M_step(U, clusters)

    #     save_picture(clusters, iteration)
    #     converge = check_converge(clusters, pre_clusters)
    
    #     iteration += 1

    return means, clusters

IMAGE_SIZE = 10000
COLOR = np.array([[56, 207, 0], [64, 70, 230], [186, 7, 61], [245, 179, 66], [55, 240, 240]])

IMAGE_ID = 1
GAMMA_S = 0.0001
GAMMA_C = 0.001
NUM_CLUSTER = 2
MODE = 'ratio'
IMAGE_PATH = f'.\data\image{IMAGE_ID}.png'
OUTPUT_DIR = f'.\output\spectral_clustering\{MODE}\image{IMAGE_ID}'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f'Image: {IMAGE_ID}, k: {NUM_CLUSTER}, mode: {MODE}')
img = np.asarray(Image.open(IMAGE_PATH).getdata()) #load image to 10000*3 numpy array

if MODE == 'ratio':
    W = kernel(img)
    L, D = compute_laplacian(W)
    U = eigen_decomposition(L)
    kmeans(U)
elif MODE == 'normalized':
    W = kernel(img)
    L, D = compute_laplacian(W)
    L_nomal = normalize_laplacian(L, D)

else:
    print('Wrong mode input!')