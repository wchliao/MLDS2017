import numpy as np
import DataPreprocessor


class DataSet(object):

    def __init__(self, datafile, dict_file):

        self._data = DataPreprocessor.ReadData(datafile)
        self._dict = DataPreprocessor.ReadDict(dict_file)

        self._datasize = len(self._data)
        self._index_in_epoch = 0
        self._N_epoch = 0

        return


    def next_batch(self, batch_size=1):
        
        x = []
        y = []

        for _ in range(batch_size):

            if self._index_in_epoch >= self._datasize:
                random_idx = np.arange(0, self._datasize)
                np.random.shuffle(random_idx)
                
                self._data = self._data[random_idx]
                
                self._index_in_epoch = 0
                self._N_epoch += 1

            x.append(self._data[self._index_in_epoch][0])
            y.append(self._data[self._index_in_epoch][1])

            self._index_in_epoch += 1

        return x, y


    @property
    def dict(self):
        return self._dict

    @property
    def datasize(self):
        return self._datasize

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def N_epoch(self):
        return self._N_epoch

