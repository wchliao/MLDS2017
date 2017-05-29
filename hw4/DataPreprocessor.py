import argparse
import os
import re
import numpy as np
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('datapaths', nargs=4, 
            help='Data paths order: movie_conversation, movie_lines, twitter_en')
    parser.add_argument('output', help='Output file name')
    parser.add_argument('dict_file', help='Dictionary file name')

    return parser.parse_args()


def line2words(line):
    words = line.lower()
    words = re.sub('[^a-z0-9? _]', '', words)
    words = words.replace('?',' ?').split()
    return words


def line2vec(line, dictionary):
    words = line.lower()
    words = re.sub('[^a-z0-9? _]', '', words)
    words = words.replace('?',' ?').split()

    vec = []
    for word in words:
        if word in dictionary:
            vec.append(dictionary[word])
        else:
            vec.append(dictionary['<UNK>'])

    return vec


def build_cnter(datapaths):
    cnter = Counter()

    with open(datapaths['movie_lines'], 'r', errors='ignore') as f:
        for line in f.readlines():
            sent = line2words(line)
            cnter.update(sent[8:])

        with open(datapaths['open_subtitles'], 'r', errors='ignore') as f:
            text = f.readlines()
            for i in range(0, len(text), 2):
                sent = line2words(text[i])
                cnter.update(sent)

        with open(datapaths['twitter'], 'r', errors='ignore') as f:
            for line in f.readlines():
                sent = line2words(line)
                cnter.update(sent)

    cnter.update('<EOS>')
    cnter.update('<BOS>')
        
    return cnter


def build_dictionary(datapaths=None, cnter=None):
    if cnter is None:
        cnter = build_cnter(datapaths)

    dictionary = {}
    for idx, wordpair in enumerate(cnter.most_common()):
        dictionary[wordpair[0]] = idx

    return dictionary, cnter


def ReadRawData(datapaths):
    output = []

    with open(datapaths['movie_lines'], 'r', errors='ignore') as f:
        lines = {}
        for line in f.readlines():
            sent = line2words(line)
            lines[sent[0]] = sent[8:]

        with open(datapaths['movie_conversations'], 'r', errors='ignore') as f:
            for line in f.readlines():
                start_idx = line.index('[')
                sents = eval(line[start_idx:])
                conversations = [lines[sent.lower()] for sent in sents]
                for i in range(len(conversations)-1):
                    output.append([conversations[i], conversations[i+1]])

        with open(datapaths['open_subtitles'], 'r', errors='ignore') as f:
            text = f.readlines()
            for i in range(0, len(text), 2):
                conversations = []
                sent = line2words(text[i])
                conversations.append(sent)
                sent = line2words(text[i+1])
                conversations.append(sent)
                output.append(conversations)

        with open(datapaths['twitter'], 'r', errors='ignore') as f:
            text = f.readlines()
            for i in range(0, len(text), 2):
                conversations = []
                sent = line2words(text[i])
                conversations.append(sent)
                sent = line2words(text[i+1])
                conversations.append(sent)
                output.append(conversations)
        
        return output


def DeNoise(cnter, freq_threshold=None, num_threshold=None):
    cut_threshold = 0
    if freq_threshold is not None:
        sort_cnter = cnter.most_common()
        for idx, wordpair in enumerate(sort_cnter):
            if wordpair[1] < freq_threshold:
                cut_threshold = idx
                break
    elif num_threshold is not None:
        sort_cnter = cnter.most_common()
        freq_threshold = sort_cnter[num_threshold-1][1]
        for idx, wordpair in enumerate(sort_cnter):
            if wordpair[1] < freq_threshold:
                cut_threshold = idx
                break
    else:
        return cnter

    new_cnter = cnter.most_common(cut_threshold)
    new_cnter = Counter(dict(new_cnter))

    if '<UNK>' not in new_cnter:
        new_cnter.update(['<UNK>'])
    if '<EOS>' not in new_cnter:
        new_cnter.update(['<EOS>'])
    if '<BOS>' not in new_cnter:
        new_cnter.update(['<BOS>'])

    return new_cnter



def str2int(data, dictionary):
    new_data = []

    for sents in data:
        new_sents = [[], []]
        for word in sents[0]:
            if word in dictionary:
                new_sents[0].append(dictionary[word])
            else:
                new_sents[0].append(dictionary['<UNK>'])
        for word in sents[1]:
            if word in dictionary:
                new_sents[1].append(dictionary[word])
            else:
                new_sents[1].append(dictionary['<UNK>'])

        new_data.append(new_sents)

    return new_data


def WriteDict(cnter, filename='dictionary.txt'):
    with open(filename, 'w') as f:
        for idx, wordpair in enumerate(cnter.most_common()):
            f.write(str(idx) + ' ' + wordpair[0] + ' ' + str(wordpair[1]) + '\n')
    return


def WriteData(data, filename):
    np.save(filename, np.array(data))
    return


def ReadDict(filename='dictionary.txt'):
    dictionary = {}
    with open(filename, 'r') as f:
        for line in f:
            s = line.split()
            dictionary[s[1]] = int(s[0])

    return dictionary


def ReadCnter(filename='dictionary.txt'):
    cnter = {}
    with open(filename, 'r') as f:
        for line in f:
            s = line.split()
            cnter[s[1]] = int(s[2])
    cnter = Counter(cnter)

    return cnter


def ReadData(filename):
    data = np.load(filename)
    return data


if __name__ == '__main__':
    args = parse_args()
    datapaths = {}
    datapaths['movie_conversations'] = args.datapaths[0]
    datapaths['movie_lines'] = args.datapaths[1]
    datapaths['open_subtitles'] = args.datapaths[2]
    datapaths['twitter'] = args.datapaths[3]

    dictionary, cnter = build_dictionary(datapaths)
    cnter = DeNoise(cnter, num_threshold=10000)
    dictionary, _ = build_dictionary(cnter=cnter)
    WriteDict(cnter, args.dict_file)
    
    data = ReadRawData(datapaths)
    data = str2int(data, dictionary)
    WriteData(data, args.output)

