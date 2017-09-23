"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 23 September, 2017 @ 7:26 PM.
  Copyright © 2017. Victor. All rights reserved.
"""
import os.path

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from features import Features

# Files and directories
DATA_DIR = 'dataset/flowers/'
LOG_DIR = 'logs/'
TENSORBOARD_DIR = os.path.join(LOG_DIR, 'tensorboard')
CHKPT_PATH = os.path.join(LOG_DIR, 'checkpoints')
MODEL_NAME = os.path.join(LOG_DIR, '5-layer-convnet.model')
# Dimensionality parameters
IMAGE_SIZE = 32
NUM_CHANNEL = 3
NUM_CLASSES = 5
# Model parameters
learning_rate = 1e-3
batch_size = 32
keep_prob = 0.8
kernel_size = 5
pool_stride = 2
hl1_depth = 8
hl2_depth = 16
hl3_depth = 32
hl4_depth = 64
hl5_depth = 128
fc1_size = 512
fc2_size = 1024

# Pre process the data
features = Features(data_dir=DATA_DIR, image_size=IMAGE_SIZE)
dataset = features.create(save=True, save_file='datasets.npy', gray=False, flatten=False)
X_train, y_train, X_test, y_test, val_X, val_y = features.train_test_split(dataset, test_size=0.1, valid_portion=0.1)

# Build the network
net = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL], name="input")
# Hidden layer 1
net = conv_2d(net, hl1_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size, pool_stride)
# Hidden layer 2
net = conv_2d(net, hl2_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size, pool_stride)
# Hidden layer 3
net = conv_2d(net, hl3_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size, pool_stride)
# Hidden layer 4
net = conv_2d(net, hl4_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size, pool_stride)
# Hidden layer 5
net = conv_2d(net, hl5_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size, pool_stride)
# Fully connected layer 1
net = fully_connected(net, fc1_size, activation='relu')
net = dropout(net, keep_prob)
# Fully connected layer 2
net = fully_connected(net, fc2_size, activation='softmax')
net = dropout(net, keep_prob)
# Output layer
net = regression(net, optimizer='adam',
                 loss='categorical_crossentropy',
                 learning_rate=learning_rate,
                 batch_size=batch_size,
                 name='output')
# Define the model (DeepNeuralNetwork)
model = tflearn.DNN(net, tensorboard_dir='logs', checkpoint_path=CHKPT_PATH)

# Check if there's a saved model
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model loaded!')
