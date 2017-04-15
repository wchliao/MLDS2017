"""

Please remember to add the following lines to restrict
the usage of GPU memory:

    gpu_options =
    tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

"""

import time
import csv
import numpy as np
import tensorflow as tf
import DataPreprocessor as DP
from DataSet import DataSet

# File names
dict_file = 'dictionary.txt'
train_file = 'train.npy'
test_file = 'test.npy'
choices_file = 'choices.npy'
model_file = './testmodel/model.ckpt'

# GPU Options
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

# Parameters
batch_size = 1
display_step = 10000
N_hidden = 256
N_epoch = 2
learning_rate = 0.001

# Constants
SPACE = 12031


def softmax(x):
    trans_x = x.transpose()
    exp_x = np.exp(trans_x)
    ret = exp_x / np.sum(exp_x, axis=0)
    return ret.transpose()


def RNN(x, w, b):
    cell = tf.contrib.rnn.BasicLSTMCell(N_hidden, forget_bias=1.0,
            state_is_tuple=True)
    rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    rnn_outputs = tf.reshape(rnn_outputs, [-1, N_hidden])
    return tf.matmul(rnn_outputs, w) + b


def run_train():
    dictionary = DP.read_dict(dict_file)
    train = DataSet(DP.read_train(train_file), len(dictionary), cut=True)

    # RNN Parameters
    N_input = train.datalen
    N_class = len(dictionary)    
    N_iter = N_epoch * N_input

    # Input
    x = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
    y = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])

    embeddings = tf.Variable(tf.random_uniform([N_class, N_hidden], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, x)

    y_reshape = tf.reshape(y, [-1])
    
    # Weights
    w = tf.Variable(tf.random_normal([N_hidden, N_class]))
    b = tf.Variable(tf.random_normal([N_class]))

    # RNN
    pred = RNN(embed, w, b)

    # cost function and optimizer
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,
        labels=y_reshape))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # accuracy
    correct_pred = tf.equal(tf.argmax(pred,1), tf.cast(y_reshape, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        step = 0

        t = time.time()
        while step < N_iter:
            batch_x, batch_y = train.next_batch(batch_size=batch_size)
            
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            
            if step % display_step == 0:
                used_time = time.time() - t
                t = time.time()
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print(str(step) + ' step: accuracy = ' + str(acc) + 
                    ' loss = '+ str(loss) + ' time = ' + str(used_time) + ' secs')
            
            step += 1

        saver = tf.train.Saver()
        saver.save(sess, model_file)
    
    return


def run_test():
    dictionary = DP.read_dict(dict_file)
    raw_test, choices = DP.read_test(test_file, choices_file)
    test = DataSet(raw_test, len(dictionary), cut=False)

    # RNN Parameters
    N_input = test.datalen
    N_class = len(dictionary)    
    N_iter = N_epoch * N_input

    # Input
    x = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
    y = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])

    embeddings = tf.Variable(tf.random_uniform([N_class, N_hidden], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, x)
    
    y_reshape = tf.reshape(y, [-1])

    # Weights
    w = tf.Variable(tf.random_normal([N_hidden, N_class]))
    b = tf.Variable(tf.random_normal([N_class]))

    # RNN
    pred = RNN(embed, w, b)

    # accuracy
    correct_pred = tf.equal(tf.argmax(pred,1), tf.cast(y_reshape, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    ans = []

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        for i in range(N_input):
            batch_x, _ = test.next_batch(batch_size=1)

            spaceID = np.argwhere(batch_x[0]==SPACE)[0,0]

            prob = sess.run(pred, feed_dict={x: batch})

            best_choice = np.argmax(prob[spaceID-1, choices[i]])
            ans.append(best_choice)
            
    return np.array(ans)


def WriteAnswer(y):
    N = len(y)
    ans = y.astype('str')

    ans[y==0] = 'a'
    ans[y==1] = 'b'
    ans[y==2] = 'c'
    ans[y==3] = 'd'
    ans[y==4] = 'e'

    with open('result.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','answer'])
        writer.writerows(np.append(np.arange(N).reshape(N,1) + 1,
            ans.reshape(N,1), axis=1))
    
    return


if __name__ == "__main__":
    t = time.time()
    ans = run_test()
    print(time.time()-t)

