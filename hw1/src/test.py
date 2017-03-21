import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import reader


class SmallConfig(object):
  """Small config."""
  n_input = 28
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 1
  n_steps = 20
  n_hidden = 200
  max_epoch = 2
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  display_step = 10
  voc_size = 12000

config = SmallConfig()

raw_data = reader.load_holmes_data(config.voc_size)
train_data, _ , _ = raw_data

class Input(object):
  """The input data."""
  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.n_steps = num_steps = config.n_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

input = Input(config, train_data)

# tf Graph input
x = tf.placeholder("float", [None, config.n_steps, config.n_input])
#y = tf.placeholder("float", [None, voc_size])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([config.n_hidden, config.voc_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([config.voc_size]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    step = 1
    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
          feed_dict[c] = state[i].c
          feed_dict[h] = state[i].h
        
        # Run optimization op (backprop)

    #sess.run(pred, feed_dict={x: input_data, y: targets})
    #    print(a)