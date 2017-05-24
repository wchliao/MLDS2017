import numpy as np
import random
from os.path import basename
import re
import json
from collections import Counter

class DataSet(object):

    def __init__(self, data, names, labels):
        self._data = data
        self._names = names
        self._labels = labels
        self._datalen = len(self._data)
        self._index_in_epoch = 0
        self._N_epoch = 0

    def next_batch(self):
        if self._index_in_epoch + 2  > self._datalen:
            self._index_in_epoch = random.randint(0, min(1, self._datalen)-1)
            self._N_epoch += 1

        start = self._index_in_epoch
        self._index_in_epoch += 1
        end = self._index_in_epoch

        return self._data[start], self._labels[start]
    
    @property
    def data(self):
        return self._data

    @property
    def datalen(self):
        return self._datalen

    @property
    def N_epoch(self):
        return self._N_epoch

    @property
    def labels(self):
        return self._labels


def build_vocabulary(captions):
    count = Counter()
    for sent in captions:
        sent = re.sub("[.!,\'\"]", "", sent)

        sent = sent.lower().split()
        count.update(sent)
    dic = {}
    dic["<PAD>"] = 0
    dic["<BOS>"] = 1
    dic["<EOS>"] = 2
    iden = 3
    for w, _ in count.most_common():
        dic[w] = iden
        iden += 1

    inv_dic = { v: k for k, v in dic.items()}
    return dic, inv_dic


def load_data(files):
    data = []
    names = []
    for f in files:
        if basename(f) == ".DS_Store": continue
        s = re.sub(".npy", "", basename(f))
        data.append(np.load(f))
        names.append(s)
    with open("../MLDS_hw2_data/training_label.json") as json_f:
        label_data = json.load(json_f)

    captions = []
    for l in label_data:
        captions += l["caption"]
    
    dic, inv_dic = build_vocabulary(captions)

    labels = []
    for row in label_data:
        row_lab = []
        for cap in row["caption"]:
            sent = re.sub("[.!,\'\"]", "", cap)
            sent = sent.lower().split()
            sent = np.array([dic[w] for w in sent])
            row_lab.append(sent)
        labels.append(row_lab)    

    return DataSet(data, names, labels)

if __name__ == "__main__":
    from os.path import join, isfile
    from os import listdir

    data_dir = "../MLDS_hw2_data"
    training_dir = join(data_dir, "training_data/feat")
    training_files = [f for f in listdir(training_dir) if isfile(join(training_dir, f))]
    training_files_full = [join(training_dir, f) for f in listdir(training_dir) if isfile(join(training_dir, f))]
    ds = load_data(training_files_full)
