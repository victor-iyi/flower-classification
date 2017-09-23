# coding: utf-8

# In[1]:
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm


# In[2]:
class Features(object):
    def __init__(self, data_dir='dataset/flowers', image_size=150):
        self.data_dir = data_dir
        self.image_size = image_size
        self.flowers = [f for f in os.listdir(self.data_dir) if f[0] is not '.']

    def create(self, save=True, save_file='datasets.npy', gray=False, flatten=True):
        datasets = []
        for flower in self.flowers:
            image_dir = os.path.join(self.data_dir, flower)
            for image_file in tqdm(os.listdir(image_dir)):
                if image_file[0] is not '.':
                    try:
                        image_path = os.path.join(image_dir, image_file)
                        img = Image.open(image_path)
                        img = img.resize((self.image_size, self.image_size))
                        if gray:
                            img = img.convert('L')
                        img = np.array(img, dtype=np.float32)
                        if flatten:
                            img = img.flatten()
                        label = self.__create_labels(flower)
                        datasets.append([img, label])
                    except Exception as e:
                        sys.stderr.write('{}'.format(e))
                        sys.stderr.flush()
        datasets = np.array(datasets)
        np.random.shuffle(datasets)
        if save:
            if os.path.isfile(save_file):
                os.unlink(save_file)
            np.save(save_file, datasets)
        return datasets

    @staticmethod
    def train_test_split(datasets, test_size=0.1):
        test_size = int(len(datasets) * test_size)
        train = datasets[:-test_size]
        test = datasets[-test_size:]
        train_X = np.array([x[0] for x in train], dtype=np.float32)
        train_y = np.array([x[1] for x in train], dtype=np.float32)
        test_X = np.array([x[0] for x in test], dtype=np.float32)
        test_y = np.array([x[1] for x in test], dtype=np.float32)
        return np.array([train_X, train_y, test_X, test_y])

    def __create_labels(self, flower):
        labels = np.zeros(len(self.flowers), dtype=np.float32)
        index = self.flowers.index(flower)
        for f in self.flowers:
            if flower == f:
                labels[index] = 1.
        return labels


# In[3]:
if __name__ == '__main__':
    features = Features()
    datasets = features.create()
    train_X, train_y, test_X, test_y = features.train_test_split(datasets)

    print('\nFirst training example:')
    print(train_X[0])
    print(train_y[0])

    print('\nFirst testing examples:')
    print(test_X[0])
    print(test_y[0])

    print('\nLength of training sets: {:,}'.format(len(train_X)))
    print('Length of testing sets:  {:,}\n'.format(len(test_X)))
