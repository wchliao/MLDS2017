import numpy as np
import random

class DataSet(object):

    def __init__(self, datapath, label, vocab_size, BOS_tag, EOS_tag):

        self._feat = []
        self._label = []
        self._maxseqlen = 0
        for y in label:
            x = np.load(datapath + y['id'] + '.npy')
            self._feat.append(x)
            self._label.append(y['caption'])

            for sent in self._label[-1]:
                sent.insert(0, BOS_tag)
                sent.append(EOS_tag)
                if len(sent) > self._maxseqlen:
                    self._maxseqlen = len(sent)

        self._datalen = len(self._label)
        self._feat_timestep = len(self._feat[0])
        self._feat_dim = len(self._feat[0][0])
        self._vocab_size = vocab_size
        self._index_in_epoch = 0
        self._N_epoch = 0

        return


    def next_batch(self, batch_size=1):
        
        x = []
        y = []

        for _ in range(batch_size):

            while self._index_in_epoch >= self._datalen:
                self._index_in_epoch = 0
                self._N_epoch += 1

            x.append(self._feat[self._index_in_epoch])
            y.append(self._label[self._index_in_epoch])

            self._index_in_epoch += 1

        return x, y


    @property
    def feat(self):
        return self._feat

    @property
    def label(self):
        return self._label

    @property
    def maxseqlen(self):
        return self._maxseqlen
    
    @property
    def datalen(self):
        return self._datalen

    @property
    def feat_timestep(self):
        return self._feat_timestep

    @property
    def feat_dim(self):
        return self._feat_dim

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def N_epoch(self):
        return self._N_epoch

