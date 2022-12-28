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
                        tmp += train_img[img][(row*3 + i) * 195 + (col*3 + j)]
                img_compress[img][row * 65 + col] = tmp // 9

    return img_compress

# def standardize(data):
#     mean = np.mean(data, axis = 0)
#     std = np.std(data, axis = 0)

#     return (data - mean) / std

def PCA(data):
    mean = np.mean(data, axis = 0)
    data_center = data - mean

    covariance = data_center @ data_center.T / data_center.shape[0]
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    eigenvector = data_center.T @ eigenvector
    
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
    for subject in trange(1, 16):
        id = np.where(label == subject)
        scatter = data[id]
        mj = np.mean(scatter, axis = 0)
        within_diff = scatter - mj
        S_w += within_diff.T @ within_diff
        between_diff = mj - m
        S_b += len(id) * between_diff.T @ between_diff
    
    S_w_S_b = np.linalg.pinv(S_w) @ S_b
    eigenvalue, eigenvector = np.linalg.eig(S_w_S_b)

    index = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, index]

    W = eigenvector[:, :25].real
    for i in range(W.shape[1]):
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

    return W

def print_eigen_fisher_face(W, face, scale):
    fig = plt.figure(figsize=(5, 5))
    for i in range(25):
        img = W[:, i].reshape(231 // scale, 195 // scale)
        ax = fig.add_subplot(5, 5, i+1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')
    fig.savefig(f'./output/{face}.jpg')
    plt.show()
        
    return

def reconstruct_face(W, data, scale):
    id = np.random.choice(135, 10, replace=False)
    fig = plt.figure(figsize=(8, 2))
    for i in range(10):
        img = data[id[i]].reshape(231 // scale, 195 // scale)
        ax = fig.add_subplot(2, 10, i + 1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')

        x = img.reshape(1, -1)
        reconstruct_img = x @ W @ W.T
        reconstruct_img = reconstruct_img.reshape(231 // scale, 195 // scale)
        ax = fig.add_subplot(2, 10, i + 11)
        ax.axis('off')
        ax.imshow(reconstruct_img, cmap='gray')
    fig.savefig(f'./output/reconstruct.jpg')
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

TRAINING_PATH = './Yale_Face_Database/Training'
TESTING_PATH = './Yale_Face_Database/Testing'

mode = int(input('Please select a mode (1~4): '))

train_img, train_filename, train_label = load_imgs(TRAINING_PATH)
test_img, test_filename, test_label = load_imgs(TESTING_PATH)

if mode == 1:
    W = PCA(train_img)
    print_eigen_fisher_face(W, 'eigenface', 1)
    reconstruct_face(W, train_img, 1)
    predict(train_img, train_label, test_img, test_label, W)
elif mode == 2:
    train_img_compress = resize_img(train_img) #77 * 65
    test_img_compress = resize_img(test_img)
    W = LDA(train_img_compress, train_label)
    print_eigen_fisher_face(W, 'fisherface', 3)
    reconstruct_face(W, train_img_compress, 3)
    predict(train_img_compress, train_label, test_img_compress, test_label, W)