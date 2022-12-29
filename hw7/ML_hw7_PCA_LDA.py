import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import trange

def read_pmg(path):
    with open(path, 'rb') as f:
        assert f.readline() == b'P5\n'
        f.readline()    #comment line
        width, height = [int(i) for i in f.readline().split()]
        assert int(f.readline()) <= 255

        img = np.zeros((height, width)) #231 * 195 = 45045
        for row in range(height):
            for col in range(width):
                img[row][col] = ord(f.read(1))

    return img.reshape(-1)

def load_imgs(path):
    image = []
    filename = []
    label = []

    for name in os.listdir(path):
        data = read_pmg(f'{path}/{name}')
        image.append(data)
        file =' '.join(name.split('.')[0:2])
        filename.append(file)
        label.append(int(file[7:9]))

    return np.array(image), np.array(filename), np.array(label)

def resize_img(data):
    num_imgs = data.shape[0]
    img_compress = np.zeros((num_imgs, 77 * 65))

    for img in range(num_imgs):
        for row in range(77):
            for col in range(65):
                tmp = 0
                for i in range(3):
                    for j in range(3):
                        tmp += data[img][(row*3 + i) * 195 + (col*3 + j)]
                img_compress[img][row * 65 + col] = tmp // 9

    return img_compress

# def standardize(data):
#     mean = np.mean(data, axis = 0)
#     std = np.std(data, axis = 0)

#     return (data - mean) / std

def PCA(data):
    covariance = np.cov(data.T)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    
    # mean = np.mean(data, axis = 0)
    # data_center = data - mean

    # covariance = data_center @ data_center.T / data_center.shape[0]
    # eigenvalue, eigenvector = np.linalg.eigh(covariance)
    # eigenvector = data_center.T @ eigenvector
    
    index = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, index]

    W = eigenvector[:, :25]
    for i in range(W.shape[1]):
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

    return W

def LDA(data, label):
    dimension = data.shape[1]
    S_w = np.zeros((dimension, dimension))
    S_b = np.zeros((dimension, dimension))
    m = np.mean(data, axis = 0)

    # scatter_mean = np.zeros((15, dimension))
    # for i in range(135):
    #     scatter_mean[label[i]-1, :] += data[i, :]
    # scatter_mean = scatter_mean / 9

    # for i in range(135):
    #     within_diff = scatter_mean[label[i]-1, :] - data[i, :]
    #     S_w += within_diff.T @ within_diff
    
    # for i in range(15):
    #     between_diff = scatter_mean[i, :] - m
    #     S_b += 9 * between_diff.T @ between_diff

    for subject in trange(1, 16):
        id = np.where(label == subject)
        scatter = data[id]
        mj = np.mean(scatter, axis = 0)
        within_diff = scatter - mj
        S_w += within_diff.T @ within_diff
        between_diff = mj - m
        S_b += len(id) * between_diff.T @ between_diff

        # xi = data[subject * 9 : (subject + 1) * 9]
        # mj = np.mean(xi, axis=0)
        # S_w += (xi - mj).T @ (xi - mj)
        # S_b += len(xi) * (mj - m).reshape(-1, 1) @ (mj - m).reshape(1, -1)

    S_w_S_b = np.linalg.pinv(S_w) @ S_b
    eigenvalue, eigenvector = np.linalg.eig(S_w_S_b)

    index = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, index]

    W = eigenvector[:, :25].real
    for i in range(W.shape[1]):
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

    return W

def linear_kernel(u, v):
    return u @ v.T

def RBFkernel(u, v):
    gamma = 1e-10
    dist = np.sum(u ** 2, axis=1).reshape(-1, 1) + np.sum(v ** 2, axis=1) - 2 * u @ v.T

    return np.exp(-gamma * dist)

def kernelPCA(data, kernel_type):
    if kernel_type == 'rbf':
        kernel = RBFkernel(data, data)
    elif kernel_type == 'linear':
        kernel = linear_kernel(data, data)
    else:
        print('False kernel type input!')
    
    eigenvalue, eigenvector = np.linalg.eigh(kernel)
    
    index = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, index]

    W = eigenvector[:, :25]
    for i in range(W.shape[1]):
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

    return W, kernel

def kernelLDA(data, kernel_type):
    Z = np.ones((data.shape[0], data.shape[0])) / 9

    if kernel_type == 'rbf':
        kernel = RBFkernel(data, data)
    elif kernel_type == 'linear':
        kernel = linear_kernel(data, data)
    else:
        print('False kernel type input!')

    S_w = kernel @ kernel
    S_b = kernel @ Z @ kernel
    
    S_w_S_b = np.linalg.pinv(S_w) @ S_b
    eigenvalue, eigenvector = np.linalg.eig(S_w_S_b)
    
    index = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, index].real

    W = eigenvector[:, :25]
    for i in range(W.shape[1]):
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

    return W, kernel

def print_eigen_fisher_face(W, face):
    fig = plt.figure(figsize=(5, 5))
    for i in range(25):
        img = W[:, i].reshape(77, 65)
        ax = fig.add_subplot(5, 5, i+1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')
    fig.savefig(f'./output/{face}.jpg')
    plt.show()
        
    return

def reconstruct_face(W, data, face):
    id = np.random.choice(135, 10, replace=False)
    fig = plt.figure(figsize=(8, 2))
    for i in range(10):
        img = data[id[i]].reshape(77, 65)
        ax = fig.add_subplot(2, 10, i + 1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')

        x = img.reshape(1, -1)
        reconstruct_img = x @ W @ W.T
        reconstruct_img = reconstruct_img.reshape(77, 65)
        ax = fig.add_subplot(2, 10, i + 11)
        ax.axis('off')
        ax.imshow(reconstruct_img, cmap='gray')
    fig.savefig(f'./output/reconstruct_{face}.jpg')
    plt.show()

    return

def predict(train_img, train_label, test_img, test_label, W):
    k = 5
    error = 0

    xW_train = train_img @ W
    xW_test = test_img @ W
    for test in range(30):
        distance = np.zeros(135)
        for train in range(135):
            distance[train] = np.sum((xW_test[test] - xW_train[train]) ** 2)
        neighbors = np.argsort(distance)[:k]
        prediction = np.argmax(np.bincount(train_label[neighbors]))
        if test_label[test] != prediction:
            error += 1

    print(f'error rate: {error / 30 * 100}%')

    return

def predict_kernel(train_img, train_label, test_img, test_label, W, train_kernel, kernel_type):
    k = 5
    error = 0

    if kernel_type == 'rbf':
        test_kernel = RBFkernel(test_img, train_img)
    elif kernel_type == 'linear':
        test_kernel = linear_kernel(test_img, train_img)
    else:
        print('False kernel type input!')

    xW_train = train_kernel @ W
    xW_test = test_kernel @ W

    for test in range(30):
        distance = np.zeros(135)
        for train in range(135):
            distance[train] = np.sum((xW_test[test] - xW_train[train]) ** 2)
        neighbors = np.argsort(distance)[:k]
        prediction = np.argmax(np.bincount(train_label[neighbors]))
        if test_label[test] != prediction:
            error += 1

    print(f'error rate: {error / 30 * 100}%')
    
    return

TRAINING_PATH = './Yale_Face_Database/Training'
TESTING_PATH = './Yale_Face_Database/Testing'

mode = int(input('Please select a mode (1~4): '))

train_img, train_filename, train_label = load_imgs(TRAINING_PATH)
test_img, test_filename, test_label = load_imgs(TESTING_PATH)

train_img_compress = resize_img(train_img) #77 * 65
test_img_compress = resize_img(test_img)

if mode == 1:   #PCA
    W = PCA(train_img_compress)
    print_eigen_fisher_face(W, 'eigenface')
    reconstruct_face(W, train_img_compress, 'eigenface')
    predict(train_img_compress, train_label, test_img_compress, test_label, W)
elif mode == 2: #LDA
    W = LDA(train_img_compress, train_label)
    print_eigen_fisher_face(W, 'fisherface')
    reconstruct_face(W, train_img_compress, 'fisherface')
    predict(train_img_compress, train_label, test_img_compress, test_label, W)
elif mode == 3: #kernel PCA
    kernel_type = 'rbf'
    mean = np.mean(train_img_compress, axis=0)
    centered_train = train_img_compress - mean
    centered_test = test_img_compress - mean
    W, train_kernel = kernelPCA(centered_train, kernel_type)
    predict_kernel(centered_train, train_label, centered_test, test_label, W, train_kernel, kernel_type)
elif mode == 4: #kernel LDA
    kernel_type = 'linear'
    mean = np.mean(train_img_compress, axis=0)
    centered_train = train_img_compress - mean
    centered_test = test_img_compress - mean
    W, train_kernel = kernelLDA(centered_train, kernel_type)
    predict_kernel(centered_train, train_label, centered_test, test_label, W, train_kernel, kernel_type)