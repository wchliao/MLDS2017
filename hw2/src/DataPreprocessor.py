"""
   Usage:  1. "python DataPreprocessor.py training_data_file" to preprocess the file
               ex: "python DataPrerpocessor.py ../data/training_label.json"
           
           2. dictionary = read_dict(dict_file) to read dictionary file
           
           3. train = read_train(data_file) to read training data file 

           4. dictionary, train = ReadAll(dict_file, train_file) to read all the files
"""

import argparse
import json
import re
import csv
import numpy as np
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess training data.')
    parser.add_argument('train_data_file', 
        help='file path for training_label.json')
    return parser.parse_args()


def build_cnter(datafile):
    cnter = Counter()

    with open(datafile, 'r') as f:
        data = json.load(f)

    for captions in data:
        for sent in captions['caption']:
            sent = sent.lower()
            sent = re.sub('[^a-z0-9 _]', '', sent)
            words = sent.split()
            cnter.update(words)

    cnter.update(['<BOS>'])
    cnter.update(['<EOS>'])

    return cnter


def build_dict(datafile=None, cnter=None):
    if cnter is None:
        cnter = build_cnter(datafile)

    dictionary = {}
    for idx, wordpair in enumerate(cnter.most_common()):
        dictionary[wordpair[0]] = idx

    return dictionary, cnter


def str2int(datafile, dictionary):

    with open(datafile, 'r') as f:
        data = json.load(f)
    
    new_data = []

    for captions in data:
        translated_captions = {}
        translated_captions['caption'] = []
        translated_captions['id'] = captions['id']

        for sent in captions['caption']:
            sent = sent.lower()
            sent = re.sub('[^a-z0-9 _]', '', sent)
            words = sent.split()

            translated_words = []
            for word in words:
                if word in dictionary:
                    translated_words.append(dictionary[word])
                else:
                    translated_words.append(dictionary['<NotExist>'])

            translated_captions['caption'].append(translated_words)
        
        new_data.append(translated_captions)

    return new_data


def write_dict(cnter, filename):
    with open(filename,'w') as f:
        for i, wordpair in enumerate(cnter.most_common()):
            f.write(str(i) + ' ' + wordpair[0] + ' ' + str(wordpair[1]) + '\n')
    
    return


def write_train(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)
    return


def read_cnter(filename):
    if filename[-3:] == 'npy':
        cnter = np.load(filename)
    else:
        cnter = {}
        with open(filename,'r') as f:
            for line in f:
                s = line.split()
                cnter[s[1]] = int(s[2])

        cnter = Counter(cnter)
        return cnter


def read_dict(filename):
    if filename[-3:] == 'npy':
        dictionary = np.load(filename)
    else:
        dictionary = {}
        with open(filename,'r') as f:
            for line in f:
                s = line.split()
                dictionary[s[1]] = int(s[0])

    return dictionary


def read_train(filename):
    if filename[-3:] == 'npy':
        data = np.load(filename)
    elif filename[-4:] == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = []

    return data


def DeNoise(cnter, freq_threshold=None, num_threshold=None):
    cut_threshold = 0
    if freq_threshold is not None:
        sort_cnter = cnter.most_common()
        for i in range(len(sort_cnter)):
            if sort_cnter[i][1] < freq_threshold:
                cut_threshold = i
                break
    elif num_threshold is not None:
        sort_cnter = cnter.most_common()
        freq_threshold = sort_cnter[num_threshold-1][1]
        for i in range(len(sort_cnter)):
            if sort_cnter[i][1] < freq_threshold:
                cut_threshold = i
                break
    else:
        return cnter, data

    new_cnter = cnter.most_common(cut_threshold-1)
    new_cnter = Counter(dict(new_cnter))
    new_cnter.update(['<NotExist>'])

    if '<BOS>' not in new_cnter:
        new_cnter.update(['<BOS>'])
    if '<EOS>' not in new_cnter:
        new_cnter.update(['<EOS>'])

    return new_cnter


def WriteAll(TrainData_path, dict_file, train_file):

    dictionary, cnter = build_dict(datafile=TrainData_path)
    cnter = DeNoise(cnter, num_threshold=3000)
    write_dict(cnter, dict_file)

    dictionary, _ = build_dict(cnter=cnter)
    train = str2int(TrainData_path, dictionary)
    write_train(train, train_file)
    
    return


def ReadAll(dict_file = 'dictionary.txt', train_file = 'train_label.json'):
    dictionary = read_dict(dict_file)
    train = read_train(train_file)

    return dictionary, train


if __name__ == "__main__":
    # Input files
    args = parse_args()

    # Output files
    dict_file = 'dictionary.txt'
    train_file = 'train_label.json'

    WriteAll(args.train_data_file, dict_file, train_file)

