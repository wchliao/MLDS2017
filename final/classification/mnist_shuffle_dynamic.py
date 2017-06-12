from __future__ import print_function  # python 2 or 3
import numpy as np
from keras.datasets import mnist
from AlexnetCNN import AlexnetCNN
from BatchSort import sorted_batches
from keras.utils import np_utils

import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def plotResults(results):
    fig, ax = plt.subplots()

    for res in results:
        history = res['history']
        label = res['label']
        ax.plot(history.accs, label = label)

    ax.set_title(r'Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    #ax.set_xscale('log')
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    fig.savefig('mnist_shuffle_dynamic_accuracy.png')

    fig, ax = plt.subplots()
    for res in results:
        history = res['history']
        label = res['label']
        ax.plot(history.val_accs, label = label)

    ax.set_title(r'Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    #ax.set_xscale('log')
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    fig.savefig('mnist_shuffle_dynamic_val_accuracy.png')

    for res in results:
        history = res['history']
        label = res['label']
        ax.plot(history.ce, label = label)

    ax.set_title(r'Cross Entropy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cross Entropy')
    #ax.set_xscale('log')
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    fig.savefig('mnist_shuffle_dynamic_ce.png')

    for res in results:
        history = res['history']
        label = res['label']
        ax.plot(history.val_ce, label = label)

    ax.set_title(r'Validation Cross Entropy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cross Entropy')
    #ax.set_xscale('log')
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    fig.savefig('mnist_shuffle_dynamic_ce.png')



def saveResults(results):
    d = {res['label'] : res['history'].ce for res in results}
    df = pd.DataFrame(d)
    df.to_csv('mnist_shuffle_dynamic_ce.csv')

    d = {res['label'] : res['history'].val_ce for res in results}
    df = pd.DataFrame(d)
    df.to_csv('mnist_shuffle_dynamic_val_ce.csv')


    d = {res['label'] : res['history'].accs for res in results}
    df = pd.DataFrame(d)
    df.to_csv('mnist_shuffle_dynamic_accuracies.csv')

    d = {res['label'] : res['history'].val_accs for res in results}
    df = pd.DataFrame(d)
    df.to_csv('mnist_shuffle_dynamic_val_accuracies.csv')


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

    #X_train, y_train = sorted_batches(X_train, y_train)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    nn = AlexnetCNN()
    nn.epochs = 20
    nn.shuffle = True
    nn.setFormat(img_rows, img_cols, img_color)
    nn.setNumClasses(nb_classes)
    nn.build_model()

    results = []
    nn.reset()
    nn.fit_dynamic(X_train, y_train, X_test, y_test)
    nn.save_weights("mnist_shuffle" + "dynamic"+ ".h5")
    results.append({ 'label': "dynamic", 
                     'history': nn.history })
    plotResults(results)
    saveResults(results)


if __name__ == "__main__":
    main()