from __future__ import print_function  # python 2 or 3
import numpy as np
from keras.datasets import mnist
from AlexnetCNN import AlexnetCNN
from BatchSort import sorted_batches
from keras.utils import np_utils
import os

import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


exp_name = "mnist_sorted"

if not os.path.exists(exp_name):
    os.makedirs(exp_name)

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
    fn = 'accuracy.png'
    file_name = os.path.join(exp_name, fn)
    fig.savefig(file_name)

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
    fn = 'val_accuracy.png'
    file_name = os.path.join(exp_name, fn)
    fig.savefig(file_name)

    for res in results:
        history = res['history']
        label = res['label']
        ax.plot(history.ce, label = label)

    ax.set_title(r'Cross Entropy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cross Entropy')
    ax.set_yscale('log')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    fn = 'ce.png'
    file_name = os.path.join(exp_name, fn)
    fig.savefig(file_name)

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
    fn = 'val_ce.png'
    file_name = os.path.join(exp_name, fn)
    fig.savefig(file_name)


def saveResultByField(results, field_name):
    d = {res['label'] : getattr(res['history'], field_name) for res in results}
    df = pd.DataFrame(d)
    fn = field_name + '.csv'
    file_name = os.path.join(exp_name, fn)
    df.to_csv(file_name)

def saveBatchResultByField(results, field_name):
    d = {res['label'] : getattr(res['history'], field_name) for res in results}
    df = pd.DataFrame(d)
    fn = 'b_' + field_name + '.csv'
    file_name = os.path.join(exp_name, fn)
    df.to_csv(file_name)

def saveResults(results):
    saveResultByField(results, 'ce')
    saveResultByField(results, 'val_ce')
    saveResultByField(results, 'accs')
    saveResultByField(results, 'val_accs')

def saveBatchResults(results):
    saveBatchResultByField(results, 'ce')
    saveBatchResultByField(results, 'val_ce')
    saveBatchResultByField(results, 'accs')
    saveBatchResultByField(results, 'val_accs')

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
    nn.epochs = 20
    nn.setFormat(img_rows, img_cols, img_color)
    nn.setNumClasses(nb_classes)
    nn.build_model()

    results = []
    batch_results = []
    for batch_size in [64, 512, 1024]:
        nn.reset()
        nn.fit(X_train, y_train, X_test, y_test, batch_size)
        fn = exp_name + "b" + str(batch_size) + ".h5"
        file_name = os.path.join(exp_name, fn)
        nn.save_weights(file_name)
        results.append({ 'label': batch_size,
                         'history': nn.history })
        batch_results.append({ 'label': batch_size,
                         'history': nn.batch_history })
    plotResults(results)
    saveResults(results)
    saveBatchResults(batch_results)

if __name__ == "__main__":
    main()