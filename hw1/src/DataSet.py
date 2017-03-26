import numpy as np
import random

class DataSet(object):

    def __init__(self, data, N_class, DataType='line'):
        self._data = np.array([])
        
        if DataType == 'batch':
            for sent in data:
                self._data = np.append(self._data, sent)
            self._data = self._data.astype(int)
        else:
            self._data = data

        self._datalen = len(self._data)
        self._N_class = N_class
        self._index_in_epoch = 0
        self._N_epoch = 0
        return


    def next_batch(self, batch_size, one_hot=True):
        if self._index_in_epoch + batch_size + 1 > self._datalen:
            self._index_in_epoch = random.randint(0, min(batch_size, self._datalen)-1)
            self._N_epoch += 1

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch

        if one_hot:
            x = np.zeros((0, self._N_class), dtype=int)
            for i in self._data[start:end+1]:
                one_hot_i = [0]*i + [1] + [0]*(self._N_class - i - 1)
                x = np.row_stack((x, one_hot_i))

            return x[:-1], x[1:]

        else:
            return self._data[start:end], self._data[end+1]


    def next_line(self, one_hot=True):
        if self._index_in_epoch == self._datalen:
            self._index_in_epoch = 0
            self._N_epoch += 1
    
        if one_hot:
            x = np.zeros((0, self._N_class), dtype=int)
            for i in self._data[self._index_in_epoch]:
                one_hot_i = [0]*i + [1] + [0]*(self._N_class - i - 1)
                x = np.row_stack((x, one_hot_i))

            return x[:-1], x[1:]

        else:
            return self._data[self._index_in_epoch][:-1],
        self._data[self._index_in_epoch][1:]
    
    @property
    def data(self):
        return self._data

    @property
    def datalen(self):
        return self._datalen

    @property
    def N_class(self):
        return self._N_class

    @property
    def N_epoch(self):
        return self._N_epoch
