import numpy as np
import DataPreprocessor


class TrainData(object):

    def __init__(self, datafile, dictfile, maxseqlen):

        self._data = DataPreprocessor.ReadData(datafile)
        self._dict = DataPreprocessor.ReadDict(dictfile)

        self._datasize = len(self._data)
        self._dictsize = len(self._dict)
        self._maxseqlen = maxseqlen

        self._index_in_epoch = 0
        self._epoch = 0

        return


    def next_batch(self, batch_size=1):
        
        x = np.full((batch_size, self._maxseqlen), self._dict['<PAD>'], dtype=np.int32)
        y = np.full((batch_size, self._maxseqlen), self._dict['<PAD>'], dtype=np.int32)
        y_seqlen = []
        y[:,0] = self._dict['<BOS>']

        for i in range(batch_size):

            if self._index_in_epoch >= self._datasize:
                self.shuffle()

            x_sent = self._data[self._index_in_epoch][0]
            y_sent = self._data[self._index_in_epoch][1]

            x[i,:len(x_sent)] = x_sent

            y[i,1:len(y_sent)+1] = y_sent
            y[i,len(y_sent)+1] = self._dict['<EOS>']

            y_seqlen.append(len(y_sent)+1)

            self._index_in_epoch += 1

        return x, y, y_seqlen


    def shuffle(self):
        random_idx = np.arange(0, self._datasize)
        np.random.shuffle(random_idx)

        self._data = self._data[random_idx]

        self._index_in_epoch = 0
        self._epoch += 1
        return


    @property
    def dict(self):
        return self._dict

    @property
    def datasize(self):
        return self._datasize

    @property
    def dictsize(self):
        return self._dictsize

    @property
    def maxseqlen(self):
        return self._maxseqlen

    @property
    def epoch(self):
        return self._epoch




class TestData(object):

    def __init__(self, datafile, dictfile, maxseqlen):


        self._data = []
        self._dict = DataPreprocessor.ReadDict(dictfile)

        with open(datafile, 'r') as f:
            for line in f.readlines():
                self._data.append(DataPreprocessor.line2vec(line, self._dict))
        
        self._datasize = len(self._data)
        self._dictsize = len(self._dict)
        self._maxseqlen = maxseqlen

        self._index = 0

        return


    def next_batch(self):
        
        x = np.full((1, self._maxseqlen), self._dict['<PAD>'], dtype=np.int32)

        if self._index >= self._datasize:
            self.shuffle()

        x_sent = self._data[self._index]
        x[0,:len(x_sent)] = x_sent

        self._index += 1
        
        y = np.full((1, self._maxseqlen), self._dict['<PAD>'], dtype=np.int32)
        y[:,0] = self._dict['<BOS>']
        
        return x, y


    def shuffle(self):
        self._index = 0
        return


    @property
    def dict(self):
        return self._dict

    @property
    def datasize(self):
        return self._datasize

    @property
    def dictsize(self):
        return self._dictsize

    @property
    def maxseqlen(self):
        return self._maxseqlen


