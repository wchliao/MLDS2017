import numpy as np

class DataSet(object):

    def __init__(self, datapath, captions, vocab_size, EOS_tag):

        self._feat = []
        self._label = []
        self._caption = []
        self._maxseqlen = 0
        
        for idx, y in enumerate(captions):
            x = np.load(datapath + y['id'] + '.npy')
            self._feat.append(x)

            for sent in y['caption']:
                self._label.append(idx)
                sent.append(EOS_tag)
                self._caption.append(sent) 
                if len(sent) > self._maxseqlen:
                    self._maxseqlen = len(sent)

        self._feat = np.array(self._feat)
        self._label = np.array(self._label)
        self._caption = np.array(self._caption)
        self._datalen = len(self._caption)
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

            if self._index_in_epoch >= self._datalen:
                random_idx = np.arange(0, self.datalen)
                np.random.shuffle(random_idx)
                
                self._label = self._label[random_idx]
                self._caption = self._caption[random_idx]
                
                self._index_in_epoch = 0
                self._N_epoch += 1

            x.append(self._feat[self._label[self._index_in_epoch]])
            y.append(self._caption[self._index_in_epoch])

            self._index_in_epoch += 1

        return np.array(x), np.array(y)


    @property
    def feat(self):
        return self._feat

    @property
    def label(self):
        return self._label

    @property
    def caption(self):
        return self._caption

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

