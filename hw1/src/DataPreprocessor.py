#
#   Usage:  1. "python DataPreprocessor.py training_data_path testing_data" to preprocess all the files
#               ex: "python DataPrerpocessor.py ../data/Holmes_Training_Data
#               testing_data.csv"
#           
#           2. dictionary = read_dict(dict_file) to read dictionary file
#           
#           3. train = read_train(data_file) to read training data file 
#
#           4. test, choices = read_test(test_file, choices_file) to read
#           testing data and choices file
#
#           5. dictionary, train, test, choices 
#               = ReadAll(dict_file, train_file, test_file, choices_file) 
#                to read all the files
#

import os
import sys
import csv
import numpy as np
import nltk
from collections import Counter

def build_cnter(datapath):
    filenames = [filename for filename in os.listdir(datapath) 
            if os.path.isfile(os.path.join(datapath, filename))]
    cnter = Counter()
    stemmer = nltk.stem.PorterStemmer()
    for filename in filenames:
        with open(os.path.join(datapath, filename), 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            for sent in nltk.sent_tokenize(raw_text):
                s = []
                for word in nltk.word_tokenize(sent):
                    try:
                        s.append(stemmer.stem(word))
                    except:
                        pass
                s = [word.lower() for word in s]
                cnter.update(s)
    if '_____' not in cnter:
        cnter.update(['_____'])

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


def str2int_train(datapath, dictionary, NotExist='NotExist'):
    filenames = [filename for filename in os.listdir(datapath) 
            if os.path.isfile(os.path.join(datapath, filename))]
    data = []
    stemmer = nltk.stem.PorterStemmer()
    for filename in filenames:
        with open(os.path.join(datapath, filename),'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            for sent in nltk.sent_tokenize(raw_text):
                words = []
                for word in nltk.word_tokenize(sent):
                    try:
                        words.append(stemmer.stem(word))
                    except:
                        pass
                words = [word.lower() for word in words]
                s = []
                for w in words:
                    if w in dictionary:
                        s.append(dictionary[w])
                    else:
                        s.append(dictionary[NotExist])

                s = np.array(s)
                data.append(s)

    return np.array(data)


def str2int_test(filename, dictionary, NotExist='NotExist'):
    data = []
    choices = []
    stemmer = nltk.stem.PorterStemmer()

    with open(filename,'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader, None)

        for line in reader:
            ID = int(line[0])
            sent = line[1]
            choice = line[2:]

            words = []
            for word in nltk.word_tokenize(sent):
                try:
                    words.append(stemmer.stem(word))
                except:
                    pass
            words = [word.lower() for word in words]
            s = []
            for w in words:
                if w in dictionary:
                    s.append(dictionary[w])
                else:
                    s.append(dictionary[NotExist])

            s = np.array(s)
            data.append(s)
            
            words = []
            for word in choice:
                try:
                    words.append(stemmer.stem(word))
                except:
                    pass
            words = [word.lower() for word in words]
            s = []
            for w in words:
                if w in dictionary:
                    s.append(dictionary[w])
                else:
                    s.append(dictionary[NotExist])

            s = np.array(s)
            choices.append(s)

    data = np.array(data)
    choices = np.array(choices)

    return data, choices


def write_dict(cnter, filename):
    cnter = cnter.most_common()
    with open(filename,'w') as f:
        for i, wordpair in enumerate(cnter):
            f.write(str(i) + ' ' + wordpair[0] + ' ' + str(wordpair[1]) + '\n')
    
    return


def write_train_file(data, filename):
    return


def write_train_npy(data, filename):
    np.save(filename, data)
    return


def write_test_file(data, choices, filename):
    return


def write_test_npy(data, choices, data_file, choices_file):
    np.save(data_file, data)
    np.save(choices_file, choices)
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
    else:
        data = []

    return data


def read_test(test_file, choices_file):
    if test_file[-3:] == 'npy':
        test = np.load(test_file)
    else:
        test = np.array([])

    if choices_file[-3:] == 'npy':
        choices = np.load(choices_file)
    else:
        choices = np.array([])

    return test, choices


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
    new_cnter.update(['NotExist'])
    if '_____' not in new_cnter:
        new_cnter.update(['_____'])

    return new_cnter


def WriteAll(TrainData_path, TestData, dict_file, train_file, test_file, choices_file):

    dictionary, cnter = build_dict(TrainData_path)
    cnter = DeNoise(cnter, None, 12000)
    write_dict(cnter, dict_file)
    dictionary, _ = build_dict(None, cnter)
    
    train = str2int_train(TrainData_path, dictionary)
    write_train_npy(train, train_file)
    
    test, choices = str2int_test(TestData, dictionary)
    write_test_npy(test, choices, test_file, choices_file)
    
    return


def ReadAll(dict_file = 'dictionary.txt', train_file = 'train.npy', 
        test_file = 'test.npy', choices_file = 'choices.npy'):
    dictionary = read_dict(dict_file)
    train = read_train(train_file)
    test, choices = read_test(test_file, choices_file)

    return dictionary, train, test, choices


if __name__ == "__main__":
    # Input files
    TrainData_path = sys.argv[1]
    TestData = sys.argv[2]

    # Output files
    dict_file = 'dictionary.txt'
    train_file = 'train.npy'
    test_file = 'test.npy'
    choices_file = 'choices.npy'

    WriteAll(TrainData_path, TestData, dict_file, train_file, test_file,
           choices_file)

