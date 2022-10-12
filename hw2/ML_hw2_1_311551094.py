import numpy as np
import math

def readOneValue(file, length):
    return int.from_bytes(file.read(length), byteorder = 'big')

def loadFile():
    training_imgs = loadImageData(TRAINING_IMAGE_FILE)
    training_labels = loadLabelData(TRAINING_LABEL_FILE)
    test_imgs = loadImageData(TEST_IMAGE_FILE)
    test_labels = loadLabelData(TEST_LABEL_FILE)

    return training_imgs, training_labels, test_imgs, test_labels

def loadImageData(filename):
    img_file = open(FILE_PATH + filename, 'rb')
    img_file.read(4)
    num_imgs = readOneValue(img_file, 4)
    num_rows = readOneValue(img_file, 4)
    num_cols = readOneValue(img_file, 4)

    imgs = []
    for _ in range(num_imgs):
        cur_img = np.zeros((num_rows, num_cols), dtype = np.uint8)
        for row in range(num_rows):
            for col in range(num_cols):
                cur_img[row][col] = readOneValue(img_file, 1)
        imgs.append(cur_img)

    img_file.close()

    return imgs

def loadLabelData(filename):
    label_file = open(FILE_PATH + filename, 'rb')
    label_file.read(4)
    num_labels = readOneValue(label_file, 4)

    labels = []
    for _ in range(num_labels):
        labels.append(readOneValue(label_file, 1))

    label_file.close()

    return labels

FILE_PATH = "./data/"
TRAINING_IMAGE_FILE = "train-images.idx3-ubyte"
TRAINING_LABEL_FILE = "train-labels.idx1-ubyte"
TEST_IMAGE_FILE = "t10k-images.idx3-ubyte"
TEST_LABEL_FILE = "t10k-labels.idx1-ubyte"

training_imgs, training_labels, test_imgs, test_labels = loadFile()
