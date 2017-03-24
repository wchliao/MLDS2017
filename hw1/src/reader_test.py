import csv
from nltk import word_tokenize


BLANKET_SYMBOL = "_____"

def _raw_questions(word_to_id=None):
	path = "../data/testing_data.csv"
	with open(path, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		next(reader, None)  # skip the headers

		for row in reader:
			qid = int(row[0])
			q = row[1]
			opts = row[2:]
			print(q)
			
			#understand_question(q)


def understand_question(q):
	q = word_tokenize(q)
	ind_blk = q.index(BLANKET_SYMBOL)
	

if __name__ == "__main__":
	_raw_questions()