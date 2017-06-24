import os
import skimage
import skimage.io
import skimage.transform
import csv
import skipthoughts
import random
import numpy as np
import re
from collections import defaultdict
import math



class DataSet(object):

    def __init__(self, imagepath, tagfile, image_size):

        self._images = []
        self._tags = []
        self._tag_vecs = []
        self._labels = []

        pattern = re.compile(r'(\D+) (hair|eyes)') 

        with open(tagfile, 'r') as f:
            print('Reading images and tags.')
            reader = csv.reader(f, delimiter=',')

            for idx, line in enumerate(reader):
                tags = line[-1]
                valid_tags = ''

                prop = {'hair': 'None', 'eyes': 'None'}

                for tag in tags.split('\t'):
                    text = tag.split(':')[0]
                    if 'hair' in text or 'eye' in text:
                        if not valid_tags:
                            valid_tags += text
                        else:
                            valid_tags += ' ' + text

                    result = pattern.match(tag)
                    if result:
                        _type = result.group(1) 
                        _attribute = result.group(2)
                        if _type in ['short', 'long']:
                            break
                        prop[_attribute] = _type

                if len(valid_tags) > 0: 
                    ID = line[0]
                    filename = os.path.join(imagepath, ID+'.jpg')
                    img = skimage.io.imread(filename)
                    img = skimage.transform.resize(img, (image_size, image_size))
                    self._images.append(img)

                    self._tags.append(valid_tags)
                    self._labels.append(prop)

        sent2vec = skipthoughts.load_model()
        self._tag_vecs = skipthoughts.encode(sent2vec, self._tags)

        self._images = np.array(self._images)
        self._tags = np.array(self._tags)
        rows = [ (x['eyes'], x['hair']) for x in self._labels]
        self._labels = np.array(rows, dtype=[('eyes', 'U9'), ('hair', 'U9')])

        self._tag_vecs = np.array(self._tag_vecs)

        self._image_num = len(self._tags)
        self._index_in_epoch = 0
        self._N_epoch = 0

        return

    def sortByProperties(self, batch_size=64):
        idx_sort = self._labels.argsort()        
        self._images = self._images[idx_sort]
        self._tags   = self._tags[idx_sort]
        self._labels = self._labels[idx_sort]

        indexDict = defaultdict(list)

        for i, y in enumerate(zip(self._labels['eyes'], self._labels['hair'])):
            indexDict[y].append(i)

        N = self._labels.shape[0]
        n_batches = int(np.floor(N / batch_size))
        idx = []
        keys_ = list(indexDict.keys())
        n_keys = len(keys_)

        indices = [0] * n_keys
        for i in range(n_batches):
            j = i % n_keys
            start = indices[j] 
            end = indices[j] + batch_size
            if end > len(indexDict[keys_[j]]):
                indices[j] = 0
            items = indexDict[keys_[j]][start:end]
            idx.extend(items)
            indices[j] += batch_size

        self._images = self._images[idx]
        self._tags   = self._tags[idx]
        self._labels = self._labels[idx]
        self._image_num = len(self._tags)



    def next_batch(self, batch_size=1):
        
        read_images = []
        wrong_images = []
        vecs = []

        for _ in range(batch_size):

            if self._index_in_epoch >= self._image_num:
                self._index_in_epoch = 0
                self._N_epoch += 1

            while True:
                random_ID = random.randint(0, self._image_num-1)
                if  self._tags[self._index_in_epoch] not in self._tags[random_ID]:
                    break

            read_images.append(self._images[self._index_in_epoch])
            wrong_images.append(self._images[random_ID])
            vecs.append(self._tag_vecs[self._index_in_epoch])

            self._index_in_epoch += 1

        return read_images, wrong_images, vecs


    @property
    def image_num(self):
        return self._image_num

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def N_epoch(self):
        return self._N_epoch

