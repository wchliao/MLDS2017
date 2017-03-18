# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import math
from nltk import word_tokenize, sent_tokenize
import numpy as np
import pickle

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab():
  data = _read_holmes()

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(word_to_id):
  data = _read_holmes()
  return [word_to_id[word] for word in data if word in word_to_id]


def _read_holmes(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  dir = "../data/Holmes_Training_Data"
  list_path = [i for i in os.listdir(dir) if os.path.isfile(os.path.join(dir, i))]
  bookname = [os.path.splitext(i)[0] for i in list_path]

  #train_books = list_path[:math.floor(len(list_path)*0.8)]
  #test_books = list_path[math.floor(len(list_path)*0.8):]

  documents = []
  for i, fn in enumerate(list_path):
      path = os.path.join(dir, fn)
      print(i, path)
      with open(path, 'r', encoding="utf-8", errors="ignore") as f:
          raw_text = f.read()
          sents = sent_tokenize(raw_text)
          for s in sents:
              tokens = [w for w in word_tokenize(s)]
              small = [w.lower() for w in tokens]
              small.append("<eos>")
              documents += small
  return documents

def holmes_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  word_to_id = _build_vocab()
  data = _file_to_word_ids(word_to_id)
  vocabulary = len(word_to_id)
  return data, vocabulary

def save_holmes_data():
  word_to_id = _build_vocab()
  data = _file_to_word_ids(word_to_id)
  vocabulary = len(word_to_id)
  
  d = (data, vocabulary)
  output = open('data.pkl', 'wb')
  pickle.dump(d, output)

  output = open('word_to_id.pkl', 'wb')
  pickle.dump(word_to_id, output)

def load_holmes_data():
  pickle_file = open('data.pkl', 'rb')
  data, vocabulary = pickle.load(pickle_file)

  pickle_file = open('word_to_id.pkl', 'rb')
  word_to_id = pickle.load(pickle_file)
  return data, vocabulary, word_to_id  

def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    #x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps], tf.ones_like([0, i * num_steps]))
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    #y = tf.strided_slice(data, [0, i * num_steps+1], [batch_size, (i + 1) * num_steps+1], tf.ones_like([0, (i+1) * num_steps+1]))
    y.set_shape([batch_size, num_steps])
    return x, y


if __name__ == "__main__":
  data, vocabulary, word_to_id  = load_holmes_data()