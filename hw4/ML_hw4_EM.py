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

    img_bin = np.zeros((num_img, num_row * num_col))
    for img in trange(num_img):
        for row in range(num_row):
            for col in range(num_col):
                img_bin[img][row * num_col + col] = 1 if imgs[img][row][col] > 127 else 0

    return img_bin

def E_step(_lamda, p, num_pixels):
    w = np.zeros((10, 2, num_pixels))   #When value 0 or 1 is shown in the speific pixel, probability for num 0~9
    indivisual_prob = np.zeros((10, 2, num_pixels)) #e.g., probability of num = 0 and value = 0 in pixel 0

    for num in range(10):
        num_chance = _lamda[num]
        for value in range(2):
            for pixel in range(num_pixels):
                indivisual_prob[num][value][pixel] = num_chance * (p[num][pixel] ** value) * ((1-p[num][pixel]) ** (1-value))
    
    for value in range(2):
        for pixel in range(num_pixels):
            marginal = sum(indivisual_prob[:, value, pixel])
            for num in range(10):
                w[num][value][pixel] = indivisual_prob[num][value][pixel] / marginal
    
    return w

FILE_PATH = "./data/"
IMAGE_FILE = "train-images.idx3-ubyte"
LABEL_FILE = "train-labels.idx1-ubyte"

imgs, labels = loadFile()
num_img= len(imgs)
num_row, num_col = imgs[0].shape

imgs = binning(imgs, num_img, num_row, num_col)

_lamda = np.full((10), 0.1)    #chance of number 0~9
p = np.random.rand(10, num_row * num_col) / 2 + 0.25 #chance of value 1 for number 0~9 and pixel 0~783, initially between 0.25~0.75
p_pre = np.zeros((10, num_row * num_col))

w = E_step(_lamda, p, num_row * num_col)