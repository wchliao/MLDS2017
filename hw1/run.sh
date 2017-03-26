wget --no-check-certificate -O data/glove.6B.zip "http://nlp.stanford.edu/data/glove.6B.zip"
unzip data/glove.6B.zip -d data
python src/cosine_similarity.py $1 $2