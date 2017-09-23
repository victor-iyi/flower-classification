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
IMAGE_SIZE = 50
NUM_CHANNEL = 3
NUM_CLASSES = 5
# Model parameters
learning_rate = 1e-3
# snapshot_step = 300
epochs = 3
batch_size = 32
keep_prob = 0.2
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
dataset = features.create(save=True, save_file='datasets.npy',
                          gray=False, flatten=False)
data = features.train_test_split(dataset, test_size=0.1, valid_portion=0.1)
X_train, y_train, X_test, y_test, X_val, y_val = data

# Build the network
net = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL],
                 name="input")
# Hidden layer 1
net = conv_2d(net, hl1_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size)
# Hidden layer 2
net = conv_2d(net, hl2_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size)
# Hidden layer 3
net = conv_2d(net, hl3_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size)
# Hidden layer 4
net = conv_2d(net, hl4_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size)
# Hidden layer 5
net = conv_2d(net, hl5_depth, kernel_size, activation='relu')
net = max_pool_2d(net, kernel_size)
# Fully connected layer 1
net = fully_connected(net, fc1_size, activation='relu')
# net = dropout(net, keep_prob)
# Fully connected layer 2
net = fully_connected(net, fc2_size, activation='relu')
net = dropout(net, keep_prob)
# Softmax layer
net = fully_connected(net, NUM_CLASSES, activation='softmax')
net = dropout(net, keep_prob)
# Output layer
net = regression(net, optimizer='adam',
                 learning_rate=learning_rate,
                 loss='categorical_crossentropy',
                 name='targets')
# Define the model (DeepNeuralNetwork)
model = tflearn.DNN(net, tensorboard_dir=TENSORBOARD_DIR,
                    checkpoint_path=CHKPT_PATH)

# Check if there's a saved model
if os.path.exists(os.path.join(LOG_DIR, '{}.meta'.format(MODEL_NAME))):
    model.load(MODEL_NAME)
    print('Model loaded!')
else:
    # Train the model
    model.fit(X_inputs={'input': X_train}, Y_targets={'targets': y_train},
              n_epoch=epochs,
              validation_set=({'input': X_test}, {'targets': y_test}),
              show_metric=True, batch_size=batch_size, run_id=MODEL_NAME)

# pred = model.predict(X_test)
# print('true =', y_test)
# print('pred =', pred)
