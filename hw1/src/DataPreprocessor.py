#
#   Usage:  1. "python DataPreprocessor.py inputpath" to preprocess all the files
#               ex: "python DataPrerpocessor.py ../data/Holmes_Training_Data"
#           
#           2. read_dict(dict_file) to read dictionary file
#           
#           3. read_data(data_file) to read data file 
#
#           4. ReadAll(dict_file, data_file) to read both dictionary and data
#              file
#

import os
import sys
import numpy as np
import nltk
from collections import Counter

def build_cnter(datapath):
    filenames = [filename for filename in os.listdir(datapath) 
            if os.path.isfile(os.path.join(datapath, filename))]
    cnter = Counter()
    for filename in filenames:
        with open(os.path.join(datapath, filename), 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            for s in nltk.sent_tokenize(raw_text):
                lower_s = [word.lower() for word in nltk.word_tokenize(s)]
                cnter.update(lower_s)

    return cnter


def build_dict(datapath, cnter=None):
    if cnter is None:
        cnter = build_cnter(datapath)

    dictionary = {}
    idx = 0
    for word in list(cnter):
        dictionary[word] = idx
        idx += 1

    return dictionary, cnter


def str2int(datapath, dictionary):
    filenames = [filename for filename in os.listdir(datapath) 
            if os.path.isfile(os.path.join(datapath, filename))]
    data = []
    for filename in filenames:
        with open(os.path.join(datapath, filename),'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            for s in nltk.sent_tokenize(raw_text):
                lower_s = [word.lower() for word in nltk.word_tokenize(s)]
                i = np.array([dictionary[word] for word in lower_s])
                data.append(i)

    return np.array(data)


def write_dict(cnter, filename):
    cnter = cnter.most_common()
    with open(filename,'w') as f:
        for i, wordpair in enumerate(cnter):
            f.write(str(i) + ' ' + wordpair[0] + ' ' + str(wordpair[1]) + '\n')
    
    return


def write_data_file(data, filename):
    return


def write_data_npy(data, filename):
    np.save(filename, data)
    return


def read_dict(filename):
    if filename[-3:] == 'npy':
        dictionary = np.load(filename)
    else:
        dictionary = {}
        with open(filename,'r') as f:
            for line in f:
                s = line.split()
                dictionary[s[1]] = s[0]

    return dictionary


def read_data(filename):
    if filename[-3:] == 'npy':
        data = np.load(filename)
    else:
        data = []
        with open(filename,'r') as f:
            for line in f:
                s = nltk.word_tokenize()
                i = np.array([int(i) for i in s])
                data.append(i)
        data = np.array(data)

    return data


def DeNoise():
    return


def WriteAll(input_path, dict_file, data_file):
    dictionary, cnter = build_dict(input_path)
    write_dict(cnter, dict_file)
    data = str2int(input_path, dictionary)
    write_data_npy(data, data_file)
    return


def ReadAll(dict_file, data_file):
    dictionary = read_dict(dict_file)
    data = read_data(data_file)
    return dictionary, data


if __name__ == "__main__":
    input_path = sys.argv[1]
    dict_file = 'dictionary.txt'
    data_file = 'data.npy'
    WriteAll(input_path, dict_file, data_file)

