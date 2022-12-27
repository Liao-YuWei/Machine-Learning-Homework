import numpy as np
import os

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

TRAINING_PATH = './Yale_Face_Database/Training'
TESTING_PATH = './Yale_Face_Database/Testing'

train_img, train_filename, train_label = load_imgs(TRAINING_PATH)
test_img, test_filename, test_label = load_imgs(TESTING_PATH)