import numpy as np
import random

class DataSet(object):

    def __init__(self, data, N_class, cut=False):

        if cut:
            self._seqlen = 50
            self._data = []
            for sent in data:
                for i in range(int(np.ceil(len(sent)/self._seqlen))):
                    self._data.append(sent[i*self._seqlen:(i+1)*self._seqlen])
            self._data = np.array(self._data)
        else:
            self._data = data
            self._seqlen = 0
            for sent in data:
                if len(sent) > self._seqlen:
                    self._seqlen = len(sent)

        self._datalen = len(self._data)
        self._N_class = N_class
        self._index_in_epoch = 0
        self._N_epoch = 0

        return


    def next_batch(self, batch_size=1):
        
        x = []
        y = []
        for _ in range(batch_size):
            while self._index_in_epoch >= self._datalen or len(self._data[self._index_in_epoch]) <= 1:
                if self._index_in_epoch >= self._datalen:
                    self._index_in_epoch = random.randint(0, batch_size)
                    self._N_epoch += 1
                elif len(self._data[self._index_in_epoch]) <= 1:
                    self._index_in_epoch += 1

            x.append(self._data[self._index_in_epoch][:-1])
            y.append(self._data[self._index_in_epoch][1:])

            self._index_in_epoch += 1

        return np.array(x), np.array(y)
   

    @property
    def data(self):
        return self._data

    @property
    def datalen(self):
        return self._datalen

    @property
    def seqlen(self):
        return self._seqlen

    @property
    def N_class(self):
        return self._N_class

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def N_epoch(self):
        return self._N_epoch

