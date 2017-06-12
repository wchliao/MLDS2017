import argparse
import os
import re
import numpy as np
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('datafiles', nargs=3,
            help='Cornell Movie Dialogue Corpus')
    parser.add_argument('output', help='Output file name')
    parser.add_argument('dictfile', help='Dictionary file name')

    return parser.parse_args()


def clean_text(text):
    cleaned_text = text.lower()
    cleaned_text = re.sub(r"\' s ", "\'s ", cleaned_text)
    cleaned_text = re.sub(r"\' m ", "\'m ", cleaned_text)
    cleaned_text = re.sub(r"\' t ", "\'t ", cleaned_text)
    cleaned_text = re.sub(r"i'm", "i am", cleaned_text)
    cleaned_text = re.sub(r"i' m", "i am", cleaned_text)
    cleaned_text = re.sub(r"he's", "he is", cleaned_text)
    cleaned_text = re.sub(r"it's", "it is", cleaned_text)
    cleaned_text = re.sub(r"that's", "that is", cleaned_text)
    cleaned_text = re.sub(r"what's", "what is", cleaned_text)
    cleaned_text = re.sub(r"where's", "where is", cleaned_text)
    cleaned_text = re.sub(r"how's", "how is", cleaned_text)
    cleaned_text = re.sub(r"how's", "how is", cleaned_text)
    cleaned_text = re.sub(r"\'ll", " will", cleaned_text)
    cleaned_text = re.sub(r"\' ll", " will", cleaned_text)
    cleaned_text = re.sub(r"\' il", " will", cleaned_text)
    cleaned_text = re.sub(r"\'ve", " have", cleaned_text)
    cleaned_text = re.sub(r"\' ve", " have", cleaned_text)
    cleaned_text = re.sub(r"\'re", " are", cleaned_text)
    cleaned_text = re.sub(r"\' re", " are", cleaned_text)
    cleaned_text = re.sub(r"\'d", " would", cleaned_text)
    cleaned_text = re.sub(r"\' d ", " would ", cleaned_text)
    cleaned_text = re.sub(r"won't", "will not", cleaned_text)
    cleaned_text = re.sub(r"can't", "cannot", cleaned_text)
    cleaned_text = re.sub(r"n't", " not", cleaned_text)
    cleaned_text = re.sub(r"n'", "ng", cleaned_text)
    cleaned_text = re.sub(r"n ' ", "ng ", cleaned_text)
    cleaned_text = re.sub(r"\'bout", "about", cleaned_text)
    cleaned_text = re.sub(r"\' bout", "about", cleaned_text)
    cleaned_text = re.sub(r"\'til", "until", cleaned_text)
    cleaned_text = re.sub(r"\' til", "until", cleaned_text)
    cleaned_text = re.sub(r"[^a-z0-9 ]", "", cleaned_text)
    return cleaned_text


def line2words(line):
    return clean_text(line).split()


def line2vec(line, dictionary):
    words = clean_text(line).split()

    vec = []
    for word in words:
        if word in dictionary:
            vec.append(dictionary[word])
        else:
            vec.append(dictionary['<UNK>'])

    return vec


def build_cnter(datafiles, maxlen=100):
    cnter = Counter()

    movie_lines = {}

    with open(datafiles['movie_lines'], 'r', errors='ignore') as f:
        for line in f.readlines():
            tokens = line.split(' +++$+++ ')
            ID = tokens[0]
            sent = line2words(tokens[-1])
            if len(sent) > maxlen or len(sent) == 0:
                continue
            else:
                movie_lines[ID] = sent

    with open(datafiles['movie_conversations'], 'r', errors='ignore') as f:
        for line in f.readlines():
            conversations = eval(line.split(' +++$+++ ')[-1])
            for i in range(len(conversations)-1):
                if conversations[i] in movie_lines and conversations[i+1] in movie_lines:
                    cnter.update(movie_lines[conversations[i]])
                    cnter.update(movie_lines[conversations[i+1]])

    with open(datafiles['open_subtitles'], 'r', errors='ignore') as f:
        for line in f.readlines():
            sent = line2words(line)
            if len(sent) > maxlen:
                continue
            cnter.update(sent)

    cnter.update(['<EOS>'])
    cnter.update(['<PAD>'])
    cnter.update(['<BOS>'])
        
    return cnter


def build_dictionary(datafile=None, cnter=None, maxlen=100):
    if cnter is None:
        cnter = build_cnter(datafile, maxlen)

    dictionary = {}
    for idx, wordpair in enumerate(cnter.most_common()):
        dictionary[wordpair[0]] = idx

    return dictionary, cnter


def ReadRawData(datafiles, maxlen=100):
    output = []
    movie_lines = {}

    with open(datafiles['movie_lines'], 'r', errors='ignore') as f:
        for line in f.readlines():
            tokens = line.split(' +++$+++ ')
            ID = tokens[0]
            sent = line2words(tokens[-1])
            if len(sent) > maxlen or len(sent) == 0:
                continue
            else:
                movie_lines[ID] = sent

    with open(datafiles['movie_conversations'], 'r', errors='ignore') as f:
        for line in f.readlines():
            conversations = eval(line.split(' +++$+++ ')[-1])
            for i in range(len(conversations)-1):
                if conversations[i] in movie_lines and conversations[i+1] in movie_lines:
                    output.append([movie_lines[conversations[i]], movie_lines[conversations[i+1]]])

    with open(datafiles['open_subtitles'], 'r', errors='ignore') as f:
        conversations = f.readlines()
        for i in range(0, len(conversations), 2):
            sent1 = line2words(conversations[i])
            sent2 = line2words(conversations[i+1])
            if len(sent1) > maxlen or len(sent) == 0:
                continue
            elif len(sent2) > maxlen or len(sent2) == 0:
                continue
            else:
                output.append([sent1, sent2])
    
    return output


def DeNoise(cnter, freq_threshold=None, num_threshold=None, keep_percent=0.97):
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
    elif keep_percent is not None:
        sort_cnter = cnter.most_common()
        all_count = 0
        remain_count = 0

        for (word, count) in sort_cnter:
            all_count += count

        for idx, (word, count) in enumerate(sort_cnter):
            remain_count += count
            if remain_count / all_count > keep_percent:
                cut_threshold = idx
                break
    else:
        return cnter

    new_cnter = cnter.most_common(cut_threshold)
    new_cnter = Counter(dict(new_cnter))

    if '<EOS>' not in new_cnter:
        new_cnter.update(['<EOS>'])
    if '<PAD>' not in new_cnter:
        new_cnter.update(['<PAD>'])
    if '<BOS>' not in new_cnter:
        new_cnter.update(['<BOS>'])
    if '<UNK>' not in new_cnter:
        new_cnter.update(['<UNK>'])

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
    datafiles = {
            'movie_conversations': args.datafiles[0],
            'movie_lines': args.datafiles[1],
            'open_subtitles': args.datafiles[2]
    }

    dictionary, cnter = build_dictionary(datafiles, maxlen=20)
    cnter = DeNoise(cnter, freq_threshold=50)
    dictionary, _ = build_dictionary(cnter=cnter)
    WriteDict(cnter, args.dictfile)
    
    data = ReadRawData(datafiles, maxlen=20)
    data = str2int(data, dictionary)
    WriteData(data, args.output)

