import tensorflow as tf
from os.path import join, isfile
from os import listdir
import numpy as np

def decode_npy(filename):
	print(filename)
	return 

training_dir = "data"
#training_dir = join(data_dir, "training_data/feat")
training_files = [join(training_dir, f) for f in listdir(training_dir) if isfile(join(training_dir, f))]
filename_queue = tf.train.string_input_producer(training_files)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

features = tf.parse_single_example(
      value,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string)
          #'label': tf.FixedLenFeature([], tf.int64),
      })

data = tf.decode_raw(features["image_raw"], tf.uint8)

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    label = sess.run([data])

  coord.request_stop()
  coord.join(threads)