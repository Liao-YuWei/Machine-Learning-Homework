import numpy as np
from tqdm import trange

def readOneValue(file, length):
    return int.from_bytes(file.read(length), byteorder = 'big')

def loadFile():
    imgs = loadImageData()
    labels = loadLabelData()

    return imgs, labels

def loadImageData():
    print('Loading', IMAGE_FILE)

    img_file = open(FILE_PATH + IMAGE_FILE, 'rb')
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

def loadLabelData():
    print('Loading', LABEL_FILE)

    label_file = open(FILE_PATH + LABEL_FILE, 'rb')
    label_file.read(4)
    num_labels = readOneValue(label_file, 4)

    labels = []
    for _ in trange(num_labels):
        labels.append(readOneValue(label_file, 1))

    label_file.close()

    return labels

def binning(imgs, num_img, num_row, num_col):
    print('Binning the image file')

    img_bin = np.zeros((num_img, num_row, num_col))
    for img in trange(num_img):
        for row in range(num_row):
            for col in range(num_col):
                img_bin[img][row][col] = 1 if imgs[img][row][col] > 127 else 0

    return img_bin

FILE_PATH = "./data/"
IMAGE_FILE = "train-images.idx3-ubyte"
LABEL_FILE = "train-labels.idx1-ubyte"

imgs, labels = loadFile()
num_img= len(imgs)
num_row, num_col = imgs[0].shape

imgs = binning(imgs, num_img, num_row, num_col)