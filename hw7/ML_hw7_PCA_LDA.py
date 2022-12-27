import numpy as np
import os
from matplotlib import pyplot as plt

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

# def standardize(data):
#     mean = np.mean(data, axis = 0)
#     std = np.std(data, axis = 0)

#     return (data - mean) / std

def PCA(data):
    mean = np.mean(data, axis = 0)
    data_center = data - mean

    covariance = data_center @ data_center.T
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    eigenvector = data_center.T @ eigenvector
    
    index = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, index]

    W = eigenvector[:, :25]
    for i in range(W.shape[1]):
        W[:, i] = W[:, i] / np.linalg.norm(W[:, i])

    return W

def print_eigen_fisher_face(W, face):
    fig = plt.figure()
    for i in range(25):
        img = W[:, i].reshape(231, 195)
        ax = fig.add_subplot(5, 5, i+1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')
    fig.savefig(f'./output/{face}.jpg')
    plt.show()
        
    return

TRAINING_PATH = './Yale_Face_Database/Training'
TESTING_PATH = './Yale_Face_Database/Testing'

train_img, train_filename, train_label = load_imgs(TRAINING_PATH)
test_img, test_filename, test_label = load_imgs(TESTING_PATH)

W = PCA(train_img)
print_eigen_fisher_face(W, 'eigenface')
# reconstruct_face(W)