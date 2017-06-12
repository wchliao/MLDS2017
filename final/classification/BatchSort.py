from keras.datasets import mnist
from collections import Counter, defaultdict
import math

def sorted_batches(X_train, y_train, batch_size=64):
    indexDict = defaultdict(list)
    for i, y in enumerate(y_train):
        indexDict[y].append(i)

    N = y_train.shape[0]
    n_batches = int(math.floor(N / batch_size))

    idx = []
    nb_classes = 10
    indices = [0] * nb_classes
    for i in range(n_batches):
        j = i % nb_classes
        start = indices[j] 
        end = indices[j] + batch_size
        if end > len(indexDict[j]):
            indices[j] = 0
        items = indexDict[j][start:end]
        idx.extend(items)
        indices[j] += batch_size
    return X_train[idx], y_train[idx]


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
    X_train, y_train = sorted_batches(X_train, y_train, 64)