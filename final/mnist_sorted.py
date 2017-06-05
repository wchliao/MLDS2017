from __future__ import print_function  # python 2 or 3
import numpy as np
from keras.datasets import mnist
from AlexnetCNN import AlexnetCNN, plotResults
from BatchSort import sorted_batches
from keras.utils import np_utils

#def sort_data(X_train, y_train):
#    idx = y_train.argsort()
#    X_train = X_train[idx]
#    y_train = y_train[idx]
#    return X_train, y_train

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    img_color, img_rows, img_cols = 1, 28, 28
    nb_classes = 10

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_color)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_color)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_train, y_train = sorted_batches(X_train, y_train)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    nn = AlexnetCNN()
    nn.setFormat(img_rows, img_cols, img_color)
    nn.setNumClasses(nb_classes)
    nn.build_model()

    results = []
    for batch_size in [32, 64, 128]:
        nn.reset()
        nn.fit(X_train, y_train, batch_size)
        results.append({ 'label': batch_size, 
                         'history': nn.history })
    plotResults(results)


if __name__ == "__main__":
    main()