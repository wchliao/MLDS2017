"""

Please remember to add the following lines to restrict
the usage of GPU memory:

    gpu_options =
    tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

"""

import argparse
import os
import json
import time
import csv
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


##### Constants #####

EOS_tag = '<EOS>'
BOS_tag = '<BOS>'
UNK_tag = '<UNK>'

#####################


##### Parameters #####

batch_size = 100
display_step = 100
N_hidden = 256
N_epoch = 10
learning_rate = 0.001
maxseqlen = 30

######################


##### GPU Options #####

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

#######################


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('testing_id_file', help='Should give testing id file')
    parser.add_argument('feature_path', help='Should give feature path here')
    
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test', action='store_true', help='Run testing')

    return parser.parse_args()


class s2sModel():
    def __init__(self, image_dim, vocab_size, N_hidden, N_video_step, N_caption_step, batch_size):
        self.image_dim = image_dim
        self.vocab_size = vocab_size
        self.N_hidden = N_hidden
        self.N_video_step = N_video_step
        self.N_caption_step = N_caption_step
        self.batch_size = batch_size

        self.word_embeddings = tf.Variable(
            tf.random_uniform([vocab_size, N_hidden], -0.1, 0.1), 
            name = 'word_embeddings')

        self.LSTM1 = tf.contrib.rnn.BasicLSTMCell(N_hidden,
                state_is_tuple=False)
        self.LSTM2 = tf.contrib.rnn.BasicLSTMCell(N_hidden,
                state_is_tuple=False)

        self.image_w = tf.Variable(
            tf.random_uniform([image_dim, N_hidden], -0.1, 0.1),
            name = 'image_w')
        self.image_b = tf.Variable(tf.zeros([N_hidden]),
            name = 'image_b')

        self.word_w = tf.Variable(
            tf.random_uniform([N_hidden, vocab_size], -0.1, 0.1),
            name = 'word_w')
        self.word_b = tf.Variable(tf.zeros([vocab_size]),
            name = 'word_b')

        return


    def build_train_model(self, dictionary):
        # Inputs
        video = tf.placeholder(dtype=tf.float32, 
                shape=[self.batch_size, self.N_video_step, self.image_dim])

        caption = tf.placeholder(dtype=tf.int32,
                shape=[self.batch_size, self.N_caption_step])
        caption_mask = tf.placeholder(dtype=tf.float32,
                shape=[self.batch_size, self.N_caption_step])

        video_flat = tf.reshape(video, [-1, self.image_dim])
        image_embed = tf.nn.xw_plus_b(video_flat, self.image_w, self.image_b)
        image_embed = tf.reshape(image_embed, 
                [self.batch_size, self.N_video_step, self.N_hidden])

        # RNN parameters
        state = tf.zeros([self.batch_size, self.LSTM1.state_size])
        padding = tf.zeros([self.batch_size, self.N_hidden])

        probs = []
        loss = 0.0

        # Encoding stage: Read frames
        for i in range(self.N_video_step):
            with tf.variable_scope('LSTM_scope') as scope:
                if i > 0:
                    scope.reuse_variables()
                
                with tf.variable_scope('LSTM1'):
                    output, state = self.LSTM1(image_embed[:,i,:], state)

        # Decoding stage: Generate captions
        for i in range(self.N_caption_step):
            if i == 0:
                cur_embed = tf.nn.embedding_lookup(self.word_embeddings,
                        np.full(self.batch_size, dictionary[BOS_tag], dtype=np.int32))
            else:
                cur_embed = tf.nn.embedding_lookup(self.word_embeddings,
                        caption[:,i-1])

            with tf.variable_scope('LSTM_scope') as scope:
                if i > 0:
                    scope.reuse_variables()

                with tf.variable_scope('LSTM2'):
                    output, state = self.LSTM2(cur_embed, state)

            logits = tf.nn.xw_plus_b(output, self.word_w, self.word_b)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    logits=logits,
                                                    labels=caption[:,i])
            cross_entropy = cross_entropy * caption_mask[:,i]

            probs.append(logits)

            cur_loss = tf.reduce_sum(cross_entropy) / self.batch_size
            loss += cur_loss

        loss = loss / tf.reduce_sum(caption_mask)

        return loss, video, caption, caption_mask, probs


    def build_test_model(self, dictionary):
        # Inputs
        video = tf.placeholder(dtype=tf.float32, 
                shape=[1, self.N_video_step, self.image_dim])
        video_flat = tf.reshape(video, [-1, self.image_dim])
        image_embed = tf.nn.xw_plus_b(video_flat, self.image_w, self.image_b)
        image_embed = tf.reshape(image_embed, [1, self.N_video_step, self.N_hidden])

        # RNN parameters
        state = tf.zeros([1, self.LSTM1.state_size])
        padding = tf.zeros([1, self.N_hidden])

        probs = []
        caption = []

        # Encoding stage: Read frames
        for i in range(self.N_video_step):
            with tf.variable_scope('LSTM_scope') as scope:
                if i > 0:
                    scope.reuse_variables()
                
                with tf.variable_scope('LSTM1'):
                    output, state = self.LSTM1(image_embed[:,i,:], state)

        # Decoding stage: Generate captions
        for i in range(self.N_caption_step):
            if i == 0:
                cur_embed = tf.nn.embedding_lookup(self.word_embeddings,
                        [dictionary[BOS_tag]])

            with tf.variable_scope('LSTM_scope') as scope:
                if i > 0:
                    scope.reuse_variables()
                
                with tf.variable_scope('LSTM2'):
                    output, state = self.LSTM2(cur_embed, state)

            logits = tf.nn.xw_plus_b(output, self.word_w, self.word_b)
            logits = tf.reshape(logits, [-1])
            logits = tf.gather(logits, np.arange(0, self.vocab_size-2))
            probs.append(logits)
            best_choice = tf.argmax(logits, axis=0)
            caption.append(best_choice)

            cur_embed = tf.nn.embedding_lookup(self.word_embeddings,
                    best_choice)
            cur_embed = tf.expand_dims(cur_embed, 0)

        return video, caption, probs


    def save_model(self, sess, model_file):
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir('model')
        saver = tf.train.Saver()
        saver.save(sess, model_file)
        return


    def restore_model(self, sess, model_file):
        if os.path.isdir(os.path.dirname(model_file)):
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
        return


def run_train():
    # Inputs
    dictionary = DP.read_dict(dict_file)
    train_label = DP.read_train(train_label_file)
    train = DataSet(train_path, train_label, len(dictionary), dictionary[EOS_tag])

    # Parameters
    N_input = train.datalen
    N_iter = N_input * N_epoch // batch_size
    print('Total training steps: %d' % N_iter)

    # Model
    model = s2sModel(
            image_dim = train.feat_dim,
            vocab_size = train.vocab_size,
            N_hidden = N_hidden,
            N_video_step = train.feat_timestep,
            N_caption_step = train.maxseqlen,
            batch_size = batch_size)

    # Loss function and optimizer
    tf_loss, tf_video, tf_caption, tf_caption_mask, _ = model.build_train_model(dictionary)
    tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

    init = tf.global_variables_initializer()

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        model.restore_model(sess, model_file)
        step = 0

        t = time.time()
        while step < N_iter:
            batch_x, batch_y = train.next_batch(batch_size=batch_size)
            y = np.full((batch_size, train.maxseqlen), dictionary[EOS_tag])
            y_mask = np.zeros(y.shape)

            for i, caption in enumerate(batch_y):
                y[i,:len(caption)] = caption
                y_mask[i, :len(caption)] = 1

            sess.run(tf_optimizer, feed_dict={
                tf_video: batch_x,
                tf_caption: y, 
                tf_caption_mask: y_mask
                })

#            if True:
            if step % display_step == 0:
                used_time = time.time() - t
                t = time.time()
                loss = sess.run(tf_loss, feed_dict={
                    tf_video: batch_x,
                    tf_caption: y,
                    tf_caption_mask: y_mask
                    })
                print(str(step) + '/' + str(N_iter) + ' step: loss = ' +
                        str(loss) + ' time = ' + str(used_time) + ' secs')
                model.save_model(sess, model_file)

            step += 1

    return


def run_test(testing_id_file, feature_path):
    # Inputs
    dictionary = DP.read_dict(dict_file)
    inv_dictionary = list(dictionary)

    ID = []
    with open(testing_id_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            ID.append(line[0])

    if feature_path[-1] is not '/':
        feature_path += '/'

    feat = []
    for filename in ID:
        x = np.load(feature_path + filename + '.npy')
        feat.append(x)
    feat = np.array(feat)

    # Parameters
    N_input = len(ID)
    feat_timestep = feat.shape[1]
    feat_dim = feat.shape[-1]
    vocab_size = len(dictionary)

    print('Total testing steps: %d' % N_input)

    # Model
    model = s2sModel(
            image_dim = feat_dim,
            vocab_size = vocab_size,
            N_hidden = N_hidden,
            N_video_step = feat_timestep,
            N_caption_step = maxseqlen,
            batch_size = batch_size)

    tf_video, tf_caption, _ = model.build_test_model(dictionary)

    init = tf.global_variables_initializer()

    result = []

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        model.restore_model(sess, model_file)
        step = 0

        t = time.time()
        for i, x in enumerate(feat):
            caption = {}
            caption['caption'] = ''
            caption['id'] = ID[i]
            pred = sess.run(tf_caption, feed_dict={tf_video: [x]})

            for j, word in enumerate(pred):
                if inv_dictionary[word] == EOS_tag:
                    break
                else:
                    if j > 0:
                        caption['caption'] += ' '

                    caption['caption'] += inv_dictionary[word]

            result.append(caption)

    return result


def WriteResult(data):
    with open('result.json','w') as f:
        json.dump(data, f, sort_keys=True, indent=4)
    return


if __name__ == '__main__':
    args = parse_args()
    if args.train:
        print('Run training.')
        run_train()
    elif args.test:
        print('Run testing.')
        result = run_test(args.testing_id_file, args.feature_path)
        WriteResult(result)

