"""

Please remember to add the following lines to restrict
the usage of GPU memory:

    gpu_options =
    tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

"""

import argparse
import json
import time
import numpy as np
import tensorflow as tf
import DataPreprocessor as DP
from DataSet import DataSet


##### Training file path #####

dict_file = 'dictionary.txt'
train_label_file = './train_label.json'
train_path = './data/training_data/feat/'
model_file = './model/model.ckpt'

##############################


##### Parameters #####

batch_size = 1
display_step = 10000
N_hidden = 256
N_epoch = 1
learning_rate = 0.001

######################


##### GPU Options #####

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

#######################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('testing_id_file', help='Should give testing id file')
    parser.add_argument('feature_path', help='Should give feature path here')
    return parser.parse_args()


class s2vtModel():
    def __init__(self, dim, vocab_size, N_hidden, N_video_step, N_caption_step, batch_size):
        self.dim = dim
        self.vocab_size = vocab_size
        self.N_hidden = N_hidden
        self.N_video_step = N_video_step
        self.N_caption_step = N_caption_step
        self.batch_size = batch_size

        self.word_embeddings = tf.Variable(
            tf.random_uniform([vocab_size, N_hidden], -0.1, 0.1), 
            name = 'word_embeddings')

        self.LSTM1 = tf.contrib.rnn.BasicLSTMCell(N_hidden)
        self.LSTM2 = tf.contrib.rnn.BasicLSTMCell(N_hidden)

        self.image_w = tf.Variable(
            tf.random_uniform([dim, N_hidden], -0.1, 0.1),
            name = 'image_w')
        self.image_b = tf.Variable(tf.zeros([N_hidden]),
            name = 'image_b')

        self.word_w = tf.Variable(
            tf.random_uniform([N_hidden, vocab_size], -0.1, 0.1),
            name = 'word_w')
        self.word_b = tf.Variable(tf.zeros([N_hidden]),
            name = 'word_b')

        return


    def build_train_model(self):
        # Inputs
        video = tf.placeholder(dtype=tf.float32, 
                shape=[self.batch_size, self.N_video_step, self.dim])

        caption = tf.placeholder(dtype=tf.int32,
                shape=[self.batch_size, self.N_caption_step])
        caption_mask = tf.placeholder(dtype=tf.float32,
                shape=[self.batch_size, self.N_caption_step])

        video_flat = tf.reshape(video, [-1, self.dim])
        image_embed = tf.nn.xw_plus_b(video_flat, self.image_w, self.image_b)
        image_embed = tf.reshape(image_embed, 
                [self.batch_size, self.N_video_step, self.N_hidden])

        # RNN parameters
        state1 = tf.zeros([self.batch_size, self.LSTM1.state_size])
        state2 = tf.zeros([self.batch_size, self.LSTM2.state_size])
        padding = tf.zeros([self.batch_size, self.N_hidden])

        probs = []
        loss = 0.0

        # Encoding stage: Read frames
        for i in range(self.N_video_step):
            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope('LSTM1'):
                output1, state1 = self.LSTM1(image_embed, state1)
            with tf.variable_scope('LSTM2'):
                output2, state2 = self.LSTM2(tf.concat([padding, output1], 1), state2)

        # Decoding stage: Generate captions
        for i in range(self.N_caption_step):
            cur_embed = tf.nn.embedding_lookup(self.word_embeddings,
                    caption[:,i])

            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('LSTM1'):
                output1, state1 = self.LSTM1(padding, state1)
            with tf.variable_scope('LSTM2'):
                output2, state2 = self.LSTM2(tf.concat([cur_embed, output1], 1), state2)

            logits = output2 * self.word_w + self.word_b
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    logits=logits,
                                                    labels=caption[:,i])
            cross_entropy = cross_entropy * caption_mask[:,i]

            probs.append(logits)

            cur_loss = tf.reduce_sum(cross_entropy)
            loss += cur_loss

        loss = loss / tf.reduce_sum(caption_mask)

        return loss, video, video_mask, caption, caption_mask, probs


    def build_test_model():
        return


def run_train():
    # Inputs
    dictionary = DP.read_dict(dict_file)
    train_label = DP.read_train(train_label_file)
    train = DataSet(train_path, train_label, len(dictionary), 
            dictionary['<BOS>'], dictionary['<EOS>'])

    # Parameters
    N_input = train.datalen
    N_iter = N_input * N_epoch

    # Model
    model = s2vtModel(
            dim = train.feat_dim,
            vocab_size = train.vocab_size,
            N_hidden = N_hidden,
            N_video_step = train.feat_timestep,
            N_caption_step = train.maxseqlen,
            batch_size = batch_size)

    # Loss function and optimizer
    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_probs = model.build_train_model()
    
    tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

    init = tf.global_variables_initializer()

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        step = 0

        t = time.time()
        while step < N_iter:
            x, raw_y = train.next_batch(batch_size=batch_size)
            y = np.array([batch_size, train.maxseqlen])
            y.fill('<EOS>')
            y_mask = np.zeros(y.shape)
            for i, caption in enumerate(raw_y):
                y[i,:len(caption)] = raw_y
                y_mask[i, :len(caption)] = 1

            sess.run(tf_optimizer, feed_dict={
                video: x,
                caption: y, 
                caption_mask: y_mask
                })

            if True:
#            if step % display_step == 0:
                used_time = time.time() - t
                t = time.time()
                loss = sess.run(tf_loss, feed_dict={
                    video: x,
                    caption: y,
                    caption_mask: y_mask
                    })
                print(str(step) + 'step: loss = ', str(loss) + 
                        ' time = ' + str(used_time) + 'secs')

            step += 1

        saver = tf.train.Saver()
        saver.save(sess, model_file)

    return


def run_test():
    return


if __name__ == '__main__':
    args = parse_args()
    run_train()

