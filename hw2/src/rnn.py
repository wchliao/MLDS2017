"""

Please remember to add the following lines to restrict
the usage of GPU memory:

    gpu_options =
    tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

"""

import numpy as np
import tensorflow as tf
import DataPreprocessor as DP
from DataSet import DataSet

# File names
dict_file = 'dictionary.txt'
train_file = 'train.npy'
test_file = 'test.npy'
choices_file = 'choice.npy'
model_file = './mymodel/model.ckpt'

# GPU Options
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# Parameters
display_step = 100
N_hidden = 256
N_epoch = 2
learning_rate = 0.001


def RNN(x, w, b, LSTM_cell):
    outputs, states = tf.nn.dynamic_rnn(LSTM_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], w)


def run_train():
    dictionary = DP.read_dict(dict_file)
    train = DataSet(DP.read_train(train_file), len(dictionary),
            DataType='line')

    # RNN Parameters
    N_input = train.datalen
    N_class = len(dictionary)    
    N_iter = N_epoch * N_input

    # Input
    x = tf.placeholder("float", [None, None, N_class])
    y = tf.placeholder("float", [None, N_class])

    # Weights
    w = tf.Variable(tf.random_normal([N_hidden, N_class]))
    b = tf.Variable(tf.random_normal([N_class]))
    LSTM_cell = tf.contrib.rnn.BasicLSTMCell(N_hidden, forget_bias=1.0,
            state_is_tuple=True)

    pred = RNN(x, w, b, LSTM_cell)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
        labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        step = 0

        while step < 10:
            batch_x, batch_y = train.next_line()
            batch_x = batch_x.reshape((1, -1, N_class))
            batch_y = batch_y.reshape((-1, N_class))

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

#            if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print(str(step) + ' step: accuracy = ' + str(acc) + 
                ' loss = '+ str(loss))
            
            step += 1

        saver = tf.train.Saver()
        saver.save(sess, model_file)


def run_test():
    dictionary = DP.read_dict(dict_file)
    train = DataSet(DP.read_train(train_file), len(dictionary),
            DataType='line')

    # RNN Parameters
    N_input = train.datalen
    N_class = len(dictionary)    
    N_iter = N_epoch * N_input

    # Input
    x = tf.placeholder("float", [None, None, N_class])
    y = tf.placeholder("float", [None, N_class])

    # Weights
    w = tf.Variable(tf.random_normal([N_hidden, N_class]))
    b = tf.Variable(tf.random_normal([N_class]))
    LSTM_cell = tf.contrib.rnn.BasicLSTMCell(N_hidden, forget_bias=1.0,
            state_is_tuple=True)

    pred = RNN(x, w, b, LSTM_cell)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
        labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        step = 0

        while step < 10:
            batch_x, batch_y = train.next_line()
            batch_x = batch_x.reshape((1, -1, N_class))

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print(str(step) + ' step: accuracy = ' + str(acc) + 
                ' loss = '+ str(loss))
            
            step += 1

    return


if __name__ == "__main__":
    run_test()

