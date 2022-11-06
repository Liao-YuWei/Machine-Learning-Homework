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
            if marginal == 0:
                continue
            for num in range(10):
                w[num][value][pixel] = indivisual_prob[num][value][pixel] / marginal
    
    return w

def M_step(imgs, w):
    num_imgs, num_pixels = imgs.shape
    _lamda = np.zeros((10))
    p = np.zeros((10, num_row * num_col))

    print('Updating lamda')
    for num in trange(10):
        sum = 0
        for img in range(num_imgs):
            for pixel in range(num_pixels):
                value = imgs[img][pixel]
                sum += w[num][value][pixel]
        sum /= (num_imgs * num_pixels)
        _lamda[num] = sum
    #print(_lamda)

    print('Updating probability to be 1 of every pixel in every label')
    for num in trange(10):
        for pixel in range(num_pixels):
            denominator = 0
            numerator = 0
            for img in range(num_imgs):
                value = imgs[img][pixel]
                denominator += w[num][value][pixel]
                if value == 1:
                    numerator += w[num][1][pixel]
            if denominator == 0:
                continue
            p[num][pixel] = numerator / denominator

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
        print('\n')
    
    return

def get_example_imgs(imgs, labels, num_pixels):
    example_imgs = np.zeros((10, num_pixels), dtype = int)
    for num in range(10):
        img_index = labels.index(num)
        example_imgs[num] = np.copy(imgs[img_index])
    
    return example_imgs

def assign_label(p, example_imgs, num_pixels):
    mapping = np.zeros(10, dtype = int)

    for num in range(10):
        same_count = np.zeros(10, dtype = int)
        for i in range(10):
            for pixel in range(num_pixels):
                pred_value = 1 if p[num][pixel] >= 0.5 else 0
                if pred_value == example_imgs[i][pixel]:
                    same_count[i] += 1
        #print(same_count)
        real_label = np.argmax(same_count)
        mapping[real_label] = num
    
    return mapping

def test(imgs, p, _lamda, mapping, num_img, num_pixel):
    prediction = np.zeros((num_img), dtype = int)

    for img in range(num_img):
        probability = np.zeros(10)
        for num in range(10):
            for pixel in range(num_pixel):
                probability[num] *= _lamda[num] * (p[num][pixel] ** imgs[img][pixel]) * \
                                    ((1-p[num][pixel]) ** (1-imgs[img][pixel]))
        pred_cluster = np.argmax(probability)
        cur_prediction = np.where(mapping == pred_cluster)
        prediction[img] = cur_prediction

    return prediction

def confusion_matrix(y, prediction, num, num_img):
    confusion = np.zeros((2, 2), dtype = int)
    for i in range(num_img):
        if y[i] == num:
            if y[i] == prediction[i]:   #TP
                confusion[0][0] += 1
            else:                       #FN
                confusion[0][1] += 1
        else:
            if y[i] == prediction[i]:   #TN
                confusion[1][1] += 1
            else:                       #FP
                confusion[1][0] += 1
    
    return confusion

def print_confusion(confusion, num):
    print(f'\nConfusion Matrix {num}:')
    print(f'\t\tPredict number {num}\tPredict not number {num}')
    print(f'Is number {num}\t\t{confusion[0][0]}\t\t\t{confusion[0][1]}')
    print(f"Isn't number {num}\t\t{confusion[1][0]}\t\t\t{confusion[1][1]}")

    sensitivity = confusion[0][0] / (confusion[0][0] + confusion[0][1])
    specificity = confusion[1][1] / (confusion[1][0] + confusion[1][1])
    print(f'\nSensitivity (Successfully predict cluster 1): {sensitivity}')
    print(f'Specificity (Successfully predict cluster 2): {specificity}')

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
    w = E_step(_lamda, p, num_row * num_col)
    _lamda, p = M_step(imgs, w)

    print_imagination(p, mapping, num_row, num_col)

    difference = sum(sum(abs(p - p_pre)))
    print(f'No. of Iteration: {iteration}, Difference: {difference}\n')
    print('------------------------------------------------------------')

    if difference < 20:
        break

    iteration += 1
    p_pre = np.copy(p)

example_imgs = get_example_imgs(imgs, labels, num_row * num_col)
mapping = assign_label(p, example_imgs, num_row * num_col)
#print(mapping)
print_imagination(p, mapping, num_row, num_col, True)

error = num_img
for num in range(10):
    prediction = test(imgs, p, _lamda, mapping, num_img, num_row * num_col)
    confusion = confusion_matrix(labels, prediction, num, num_img)
    num_correct = print_confusion(confusion, num)
    error -= num_correct
error /= num_img

print(f'Total iteration to converge: {iteration}')
print(f'Total error rate: {error}')

sys.stdout = original_stdout