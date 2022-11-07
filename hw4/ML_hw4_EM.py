import numpy as np
from tqdm import trange
import sys

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

    img_bin = np.zeros((num_img, num_row * num_col), dtype = int)
    for img in trange(num_img):
        for row in range(num_row):
            for col in range(num_col):
                img_bin[img][row * num_col + col] = 1 if imgs[img][row][col] > 127 else 0

    return img_bin

def E_step(imgs, _lamda, p, num_imgs, num_pixels):
    w = np.zeros((num_img, 10)) #For every specific image, its probability to be cluster 0~9

    print('E step')
    for img in trange(num_imgs):
        for cluster in range(10):
            w[img][cluster] = _lamda[cluster]
            for pixel in range(num_pixels):
                if imgs[img][pixel] == 1:
                    w[img][cluster] *= p[cluster][pixel]
                else:
                    w[img][cluster] *= (1 - p[cluster][pixel])
        if sum(w[img]) == 0:
            continue
        w[img] /= sum(w[img])
    
    return w

def M_step(imgs, w):
    num_imgs, num_pixels = imgs.shape
    _lamda = np.zeros((10))
    p = np.zeros((10, num_row * num_col))

    print('M step')
    for cluster in trange(10):
        _lamda[cluster] = sum(w[:, cluster]) / num_imgs
        for pixel in range(num_pixels):
            dot = 0
            for img in range(num_imgs):
                if imgs[img][pixel] == 1:
                    dot +=  w[img][cluster]
            if _lamda[cluster] == 0:
                continue
            p[cluster][pixel] = dot / (_lamda[cluster] * num_imgs)

    return _lamda, p

def print_imagination(p, mapping, num_row, num_col, labeled = False):
    for i in range(10):
        if labeled:
            print("Labeled", end=" ")
        print(f'Class {i}:')
        index = int(mapping[i])
        for row in range(num_row):
            for col in range(num_col):
                pixel = 1 if p[index][row * num_row + col] >= 0.5 else 0
                print(pixel, end = ' ')
            print('')
        print('')
    
    return

def assign_label_and_test(w, labels, num_imgs):
    mapping = np.zeros((10), dtype = int)
    counting = np.zeros((10, 10), dtype = int)
    prediction = np.zeros((num_img), dtype = int)

    #Cluster the images and count the relation between real label and each cluster
    print('Clustering images')
    for img in trange(num_imgs):
        predict_cluster = np.argmax(w[img])
        counting[labels[img], predict_cluster] += 1
        prediction[img] = predict_cluster
    
    #Use array mapping to store the corresponding cluster of the specific label
    print('Finding the real label for each cluster')
    for _ in trange(10):
        index = np.argmax(counting)
        real_label = index // 10
        cur_cluster = index % 10
        mapping[real_label] = cur_cluster
        counting[real_label, :] = 0
        counting[:, cur_cluster] = 0
    
    #Modify the prediction from cluster to real label
    print('Mapping the real label for each image prediciton')
    mapping_inv = np.zeros((10), dtype = int)
    for i in range(10):
        mapping_cluster = mapping[i]
        mapping_inv[mapping_cluster] = i
    for img in trange(num_imgs):
        predict_cluster = prediction[img]
        prediction[img] = mapping_inv[predict_cluster]

    return mapping, prediction

def confusion_matrix(y, prediction, num, num_img):
    confusion = np.zeros((2, 2), dtype = int)
    for i in range(num_img):
        if y[i] == num:
            if prediction[i] == num:    #TP
                confusion[0][0] += 1
            else:                       #FN
                confusion[0][1] += 1
        else:
            if prediction[i] != num:    #TN
                confusion[1][1] += 1
            else:                       #FP
                confusion[1][0] += 1
    
    return confusion

def print_confusion(confusion, num):
    print(f'\nConfusion Matrix {num}:')
    print(f'\t\t\t\tPredict number {num}\tPredict not number {num}')
    print(f'Is number {num}\t\t\t\t{confusion[0][0]}\t\t\t{confusion[0][1]}')
    print(f"Isn't number {num}\t\t\t{confusion[1][0]}\t\t\t{confusion[1][1]}")

    sensitivity = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    specificity = confusion[1][1] / (confusion[1][0] + confusion[1][1])
    print(f'\nSensitivity (Successfully predict number {num}): {sensitivity}')
    print(f'Specificity (Successfully predict not number {num}): {specificity}')

    return confusion[0][0]

original_stdout = sys.stdout
f = open('EM_result.txt', 'w')
sys.stdout = f

FILE_PATH = "./data/"
IMAGE_FILE = "train-images.idx3-ubyte"
LABEL_FILE = "train-labels.idx1-ubyte"
MAX_ITERATION = 50

imgs, labels = loadFile()
num_img= len(imgs)
num_row, num_col = imgs[0].shape

imgs = binning(imgs, num_img, num_row, num_col)

_lamda = np.full((10), 0.1)    #chance of number 0~9
p = np.random.rand(10, num_row * num_col) / 2 + 0.25 #chance of value 1 for number 0~9 and pixel 0~783, initially between 0.25~0.75
p_pre = np.zeros((10, num_row * num_col))
mapping = np.arange(10)
iteration = 1

while iteration < MAX_ITERATION:
    w = E_step(imgs, _lamda, p, num_img, num_row * num_col)
    _lamda, p = M_step(imgs, w)

    print_imagination(p, mapping, num_row, num_col)

    difference = sum(sum(abs(p - p_pre)))
    print(f'No. of Iteration: {iteration}, Difference: {difference}\n')
    print('------------------------------------------------------------')

    if difference < 20:
        break

    iteration += 1
    p_pre = np.copy(p)

mapping, prediction = assign_label_and_test(w, labels, num_img)
print_imagination(p, mapping, num_row, num_col, True)

error = num_img
for num in range(10):
    confusion = confusion_matrix(labels, prediction, num, num_img)
    num_correct = print_confusion(confusion, num)
    error -= num_correct
error /= num_img

print(f'\nTotal iteration to converge: {iteration}')
print(f'Total error rate: {error}')

sys.stdout = original_stdout