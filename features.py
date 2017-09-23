"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 23 September, 2017 @ 7:06 PM.
  Copyright © 2017. Victor. All rights reserved.
"""

# coding: utf-8

import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm


class Features(object):
    def __init__(self, data_dir='dataset/flowers', image_size=32):
        """
        Data pre-processing for flower classification.

        :param data_dir:
                Where the dataset is located.
        :param image_size:
                Size to resize images into.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.flowers = [f for f in os.listdir(self.data_dir) if f[0] is not '.']

    def create(self, save=True, save_file='datasets.npy', gray=False, flatten=True):
        """
        Create dataset and optionally saves it into a numpy file.

        :param save: bool, default True (optional)
                If save it set to false, the dataset will not be saved
                and you need to rerun this function again to build dataset
        :param save_file:
                This defines where to save the dataset after it has been built.
        :param gray: bool, default False (optional)
                Convert images to grayscale. This is done to reduce the number
                of color channels the image has.
        :param flatten: bool, default True (optional)
                Collapse the image into a single dimension or 1-D. This is most
                useful for feed-forward networks but BAD for Convolutional
                Neural Networks.
        :return: datasets: `np.ndarray` consisting of `np.array` of "features" and "labels".
        """
        if os.path.isfile(save_file):
            return np.load(save_file)
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
                os.unlink(save_file)  # removes previously saved file
            np.save(save_file, datasets)
        return datasets

    @staticmethod
    def train_test_split(dataset, test_size=0.1, **kwargs):
        """
        Splits dataset into training and testing set.

        :param dataset: Dataset to be split.
        :param test_size: float, default 0.1
                    Size of the testing data in %.
                    Default is 0.1 or 10% of the dataset.
        :keyword valid_portion: float
                    Size of validation set in %.
                    This will be taking from training set
                    after splitting into training and testing set.
        :return: np.array of train_X, train_y, test_X, test_y
        """
        test_size = int(len(dataset) * test_size)

        train = dataset[:-test_size]
        test = dataset[-test_size:]

        train_X = np.array([x[0] for x in train], dtype=np.float32)
        train_y = np.array([x[1] for x in train], dtype=np.float32)
        test_X = np.array([x[0] for x in test], dtype=np.float32)
        test_y = np.array([x[1] for x in test], dtype=np.float32)

        if 'valid_portion' in kwargs:
            valid_portion = kwargs['valid_portion']
            valid_portion = int(len(train) * valid_portion)

            train = train[:-valid_portion]
            val = train[-valid_portion:]

            val_X = np.array([x[0] for x in val], dtype=np.float32)
            val_y = np.array([x[1] for x in val], dtype=np.float32)

            return np.array([train_X, train_y, test_X, test_y, val_X, val_y])

        return np.array([train_X, train_y, test_X, test_y])

    def __create_labels(self, flower):
        labels = np.zeros(len(self.flowers), dtype=np.float32)
        index = self.flowers.index(flower)
        for f in self.flowers:
            if flower == f:
                labels[index] = 1.
        return labels


if __name__ == '__main__':
    features = Features()
    datasets = features.create()
    train_X, train_y, test_X, test_y, val_X, val_y = features.train_test_split(datasets, valid_portion=0.1)

    print('Training set ->', train_X.shape, test_X.shape, val_X.shape)
    print('Testing set ->', train_y.shape, test_y.shape, val_y.shape)

    print('\nLength of training sets:\t\t{:,}'.format(len(train_y)))
    print('Length of testing sets: \t\t{:,}'.format(len(test_y)))
    print('Length of validation sets:  {:,}\n'.format(len(val_y)))
