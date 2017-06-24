from __future__ import print_function  # python 2 or 3
import numpy as np
import random
#import cPickle as pickle
import argparse

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

from keras.layers import Input, RepeatVector, Permute, Reshape, BatchNormalization, Lambda, K
from keras.models import Model
from keras.regularizers import l2
#from tqdm import tqdm
from keras.callbacks import EarlyStopping, Callback
from keras.datasets import mnist
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
set_session(tf.Session(config=config))

# MNIST
nb_classes = 10 # number of categories we classify. MNIST is 10 digits
    # input image dimensions. In CNN we think we have a "color" image with 1 channel of color.
    # in MLP with flatten the pixels to img_rows*img_cols
img_color, img_rows, img_cols = 1, 28, 28
img_size = img_color * img_rows * img_cols
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
nhiddens = [512]
opt = 'adam' 

DROPOUT=0.5
weight_decay = None

batch_size = 128
epochs = 1
FN = "mnist"

regularizer = l2(weight_decay) if weight_decay else None

basic_model = Sequential(name='basic')

basic_model.add(Conv2D(nb_filters, nb_conv, nb_conv,
                                 border_mode='valid',
                                 input_shape=(img_rows, img_cols, img_color)))
basic_model.add(Activation('relu'))
basic_model.add(Conv2D(nb_filters, nb_conv, nb_conv))
basic_model.add(Activation('relu'))
basic_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
basic_model.add(Dropout(0.25))

basic_model.add(Conv2D(nb_filters*2, nb_conv, nb_conv, border_mode='same'))
basic_model.add(Activation('relu'))
basic_model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
basic_model.add(Activation('relu'))
basic_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
basic_model.add(Dropout(0.25))

basic_model.add(Flatten())
for nhidden in nhiddens:
    basic_model.add(Dense(nhidden, W_regularizer=regularizer))
    basic_model.add(Activation('relu'))
    basic_model.add(Dropout(DROPOUT))

basic_model.add(Dense(nb_classes, activation='softmax',
                     name='basic_dense'))

basic_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_color)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_color)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

basic_model.fit(X_train,
       y_train,
       validation_data=(X_test, y_test),
       batch_size=batch_size, nb_epoch=epochs,
       callbacks=[history])

score = basic_model.evaluate(X_test, y_test)
print()
print(score)

# summarize history for accuracy
plt.plot(history.losses)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
