import numpy as np
import csv
import re
from scipy import spatial
import operator

# load input data
testFile = '../data/testing_data.csv'
sentences = []
choices = []
print "Loading Test Data"
with open(testFile,'r') as fin:
    reader = csv.reader(fin,delimiter=',')
    for row in reader:
        sentences.append(row[1])
        t = []
        t.append(row[2])
        t.append(row[3])
        t.append(row[4])
        t.append(row[5])
        t.append(row[6])
        choices.append(t)
del sentences[0]
del choices[0]
print "Done.",len(sentences)," data loaded"

def getWords(text):
    a = re.compile('\w+').findall(text)
    a = [x.lower() for x in a]
    return a

# preprocess input data
words = []
for s in sentences:
    words.append(getWords(s))

# load word embeddings
##############################################
gloveFile = '../data/glove.6B.300d.txt'
##############################################
vocab = {}
print "Loading Glove Model"
with open(gloveFile,'r') as fin:
    for line in fin:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        vocab[word] = embedding
print "Done.",len(vocab)," words loaded"

# make prediction
pred = []
window = 10
for i in range(len(sentences)):
    blank_idx = words[i].index('_____')
    vector_sum = np.zeros(300)
    # calculate vector sums
    for j in range(blank_idx-window, blank_idx+window):
        if (j>=0) and (j!=blank_idx) and (j<len(words[i])):
            if words[i][j] in vocab:
                vector_sum = vector_sum + vocab[words[i][j]]
    similarity = []
    # calculate similarity with each choice
    for j in range(5):
        if choices[i][j] in vocab:
            ans = 1 - spatial.distance.cosine(vector_sum, vocab[choices[i][j]])
            similarity.append(ans)
        else:
            ans = 1 - spatial.distance.cosine(vector_sum, np.zeros(300))
            similarity.append(ans)
    # get max similarity
    m_index, m_value = max(enumerate(similarity), key=operator.itemgetter(1))
    pred.append(m_index)

# output prediction
predFile = 'pred.csv'
ch = ['a','b','c','d','e']
with open(predFile,'w') as fout:
    fout.write('id,answer')
    fout.write('\n')
    for i in range(len(pred)):
        fout.write(str(i+1)+','+str(ch[pred[i]]))
        fout.write('\n')





