import numpy as np
from collections import namedtuple
from sklearn.cluster import AgglomerativeClustering

def loadGloveModel(gloveFile, d=25):
    print("Loading Glove Model")
    model = {"vocabulary": [], "X": []}
    
    word_vecs = []
    N = 0

    with open(gloveFile, encoding='UTF-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            if len(embedding) < d: continue
            model[word] = embedding
            N += 1

    print("Done.",len(model)," words loaded!")
    return model

def loadVocabulary(vfile):
    print("Loading Vocabulary")
    voc = set()
    with open(v_file) as f:
        for line in f:
            _id, w = line.split()
            voc.add(w)
    return voc

def mergeWordVectors(model, vocabulary):
    vecs = []
    n_oov = 0
    for v in vocabulary:
        if not v in model: 
            vecs.append(np.random.rand(1,25))
            n_oov += 1
        else: vecs.append(model[v])
    X = np.array(vecs)
    return X

def cluster(X):
    clustering = AgglomerativeClustering(linkage='average')
    clustering.fit(X)
    np.savetxt("children.csv", clustering.children_)
    return clustering

def saveVocabulary(model):
    for i, x in enumerate(model["vocabulary"]):
        print(i, x)

if __name__ == "__main__":
    path = "/data/fredericgo/glove/glove.twitter.27B.25d.txt"
    v_file = "vocabulary.csv"
    model = loadGloveModel(path)
    vocabulary = loadVocabulary(v_file)
    X = mergeWordVectors(model, vocabulary)
    clustering = cluster(model)
    #saveVocabulary(model)
