from __future__ import division
import nltk, re, pprint
from nltk import word_tokenize, sent_tokenize
from collections import Counter
import os
import numpy as np
import csv
import pickle

dir = "../Holmes_Training_Data"
list_path = [i for i in os.listdir(dir) if os.path.isfile(os.path.join(dir, i))]
bookname = [os.path.splitext(i)[0] for i in list_path]

test_fn = os.path.join(dir, list_path[0])

print(len(list_path))
vocabulary = set()
counter = Counter()

for i, fn in enumerate(list_path):
    path = os.path.join(dir, fn)
    print(i, path)
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
        tokens = word_tokenize(raw_text)
        small = [w.lower() for w in tokens]
        counter.update(small)
    if i > 10: break

word_ids = {}
with open('vocabulary.csv', 'w') as csvfile:
    wid = 0
    spamwriter = csv.writer(csvfile, delimiter=' ')
    for w, f in counter.most_common():
        spamwriter.writerow([wid, w, f])
        word_ids[w] = wid
        wid += 1

documents = []
for i, fn in enumerate(list_path):
    path = os.path.join(dir, fn)
    print(i, path)
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
        #tokens = word_tokenize(raw_text)
        sents = sent_tokenize(raw_text)
        for s in sents:
            tokens = [w for w in word_tokenize(s)]
            small = [w.lower() for w in tokens]
            nums = np.array([word_ids[w] for w in small])
            documents.append(nums)
    #if i > 10: break
documents = np.array(documents)

np.save("documents.npy", documents)

