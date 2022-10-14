import numpy as np
import math
from tqdm import trange

def readOneValue(file, length):
    return int.from_bytes(file.read(length), byteorder = 'big')

def loadFile():
    training_imgs = loadImageData(TRAINING_IMAGE_FILE)
    training_labels = loadLabelData(TRAINING_LABEL_FILE)
    test_imgs = loadImageData(TEST_IMAGE_FILE)
    test_labels = loadLabelData(TEST_LABEL_FILE)

    return training_imgs, training_labels, test_imgs, test_labels

def loadImageData(filename):
    print('Loading', filename)

    img_file = open(FILE_PATH + filename, 'rb')
    img_file.read(4)
    num_imgs = readOneValue(img_file, 4)
    num_rows = readOneValue(img_file, 4)
    num_cols = readOneValue(img_file, 4)

    imgs = []
    for _ in trange(num_imgs):
        cur_img = np.zeros((num_rows, num_cols), dtype = np.uint8)
        for row in range(num_rows):
            for col in range(num_cols):
                cur_img[row][col] = readOneValue(img_file, 1)
        imgs.append(cur_img)

    img_file.close()
    
    return imgs

def loadLabelData(filename):
    print('Loading', filename)

    label_file = open(FILE_PATH + filename, 'rb')
    label_file.read(4)
    num_labels = readOneValue(label_file, 4)

    labels = []
    for _ in trange(num_labels):
        labels.append(readOneValue(label_file, 1))

    label_file.close()

    return labels

def printResult(posterior, label):
    print('Posterior (in log scale):')
    for i in range(10):
        print(f'{i}: {posterior[i]}')
    prediction = np.argmin(posterior)
    print(f'Prediction: {prediction}, Ans: {label}')
    
    return 0 if prediction == label else 1


FILE_PATH = "./data/"
TRAINING_IMAGE_FILE = "train-images.idx3-ubyte"
TRAINING_LABEL_FILE = "train-labels.idx1-ubyte"
TEST_IMAGE_FILE = "t10k-images.idx3-ubyte"
TEST_LABEL_FILE = "t10k-labels.idx1-ubyte"

training_imgs, training_labels, test_imgs, test_labels = loadFile()
num_train, num_test = len(training_imgs), len(test_imgs)
num_row, num_col = training_imgs[0].shape

prior = np.zeros(10)
likelihood = np.zeros((10, num_row*num_col, 32))

for i in trange(num_train):
    label = training_labels[i]
    prior[label] += 1
    for row in range(num_row):
        for col in range(num_col):
            bin = training_imgs[i][row][col] // 8
            likelihood[label][row*num_col + col][bin] += 1

for label in trange(10):
    num_label = prior[label]
    for i in range(num_row * num_col):
        for bin in range(32):
            if likelihood[label][i][bin] != 0:
                likelihood[label][i][bin] /= num_label
            else:
                likelihood[label][i][bin] = 0.000000001
prior /= num_train

error = 0
for i in range(num_test):
    posterior = np.zeros(10)
    for num in range(10):
        posterior[num] = math.log(prior[num])
        for row in range(num_row):
            for col in range(num_col):
                cur_px_bin = test_imgs[i][row][col] // 8
                posterior[num] += math.log(likelihood[num][row*num_col + col][cur_px_bin])
    posterior /= sum(posterior)
    error += printResult(posterior, test_labels[i])
error /= num_test

print(f'Error rate: {error}')


