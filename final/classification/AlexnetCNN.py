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

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.accs = []
        self.val_accs = []

    def on_epoch_end(self, batch, logs={}):
        self.accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

class AlexnetCNN(object):
    def __init__(self):
        self.nb_filters = 32
        self.nb_pool = 2
        self.nb_conv = 3
        self.nhiddens = [512]
        self.opt = 'adam' 

        self.DROPOUT = 0.5
        self.weight_decay = None

        self._batch_size = 128
        self._epochs = 5

        self.regularizer = l2(self.weight_decay) if self.weight_decay else None

        self._shuffle = False

        self.history = LossHistory()

    def setFormat(self, img_rows, img_cols, img_color):
        self.img_color, self.img_rows, self.img_cols = img_color, img_rows, img_cols

    def setNumClasses(self, nb_classes):
        self.nb_classes = nb_classes

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        self._shuffle = shuffle

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        self._epochs = epochs

    @property
    def batch_size(self):
        return self._batch_size 

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size


    def build_model(self):
        model = Sequential(name='basic')

        model.add(Conv2D(self.nb_filters, self.nb_conv, self.nb_conv,
                                         border_mode='valid',
                                         input_shape=(self.img_rows, self.img_cols, self.img_color)))
        model.add(Activation('relu'))
        model.add(Conv2D(self.nb_filters, self.nb_conv, self.nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool)))
        model.add(Dropout(0.25))

        model.add(Conv2D(self.nb_filters*2, self.nb_conv, self.nb_conv, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.nb_filters*2, self.nb_conv, self.nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        for nhidden in self.nhiddens:
            model.add(Dense(nhidden, W_regularizer=self.regularizer))
            model.add(Activation('relu'))
            model.add(Dropout(self.DROPOUT))

        model.add(Dense(self.nb_classes, activation='softmax',
                             name='basic_dense'))

        model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        self.model = model
        self._weights = self.model.get_weights()

    def fit(self, X_train, y_train, X_val, y_val, batch_size = 128):
        self.model.fit( X_train, y_train, 
                        batch_size = batch_size, 
                        nb_epoch = self._epochs,
                        validation_data = (X_val, y_val),
                        callbacks=[self.history])

    def reset(self):
        self.model.set_weights(self._weights)
        self.history = LossHistory()



if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img_color, img_rows, img_cols = 1, 28, 28
    nb_classes = 10

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_color)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_color)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    nn = AlexnetCNN()
    nn.setFormat(img_rows, img_cols, img_color)
    nn.setNumClasses(nb_classes)
    nn.build_model()

    results = []
    for batch_size in [32, 64, 128]:
        nn.reset()
        nn.fit(X_train, y_train, X_test, y_test, batch_size)
        results.append({ 'label': batch_size, 
                         'history': nn.history })
    plotResults(results)



