# Homework 4 

The following are the usages of the code.
You can also check `run.sh` for the example of the usage.


## Data Preprocessor

### Usage

```bash
$ python DataPreprocessor.py [datafile] [outputfile] [dictfile]
```

* `[datafile]`: the path of `open_subtitles.txt`
* `[outputfile]`: the path of the output file you want to create
* `[dictfile]`: the path of the dictionary you want to create


## Seq2Seq

### Usage (Training)

```bash
$ python seq2seq.py --train [-t datafile] [-d dictionary]
```

* `[-t datafile]`: the path of preprocessed data
* `[-d dictionary]`: the path of the dictionary

### Usage (Testing)

```bash
$ python seq2seq.py --test [-q input] [-o output] [-d dictionary]
```

* `[-q input]`: the path of input file (ex. `question.txt`)
* `[-o output]`: the path of the output file you want to create
* `[-d dictionary]`: the path of the dictionary

