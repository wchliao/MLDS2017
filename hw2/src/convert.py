"""Converts MNIST data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

from os.path import join, isfile, basename
from os import listdir
import numpy as np
import re

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))


def convert_to(data_set, name):
  """Converts a dataset to tfrecords."""
  data = data_set.data

  length = data.shape[0]
  dim    = data.shape[1]

  f = re.sub(".npy", "", basename(name))
  filename = os.path.join("data", f + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(length):
    data_raw = data[index]

    example = tf.train.Example(features=tf.train.Features(feature={
        #'label': _int64_feature(int(labels[index])),
        'data_raw': _float_feature(data_raw)}))
    writer.write(example.SerializeToString())
  writer.close()


class Dataset:
  pass

def main(unused_argv):
  # Get the data.
  data_dir = "../MLDS_hw2_data"
  training_dir = join(data_dir, "training_data/feat")
  training_files = [f for f in listdir(training_dir) if isfile(join(training_dir, f)) and f != ".DS_Store"]
  training_full = [join(training_dir, f) for f in training_files]
  ds = Dataset()
  for f in training_full:
    x = np.load(f)
    ds.data = x
    convert_to(ds, f)
  

if __name__ == '__main__':
  tf.app.run(main=main)