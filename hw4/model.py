import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
import utils
import os


class seq2seqModel(object):
    def __init__(self, batch_size, vocab_size, maxseqlen=30, hidden_dim=256):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.maxseqlen = maxseqlen

        self.proj_w = tf.get_variable('proj_w', [vocab_size, hidden_dim],
                initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self.proj_b = tf.get_variable('prob_b', [vocab_size], 
                initializer=tf.zeros_initializer())

        self.LSTM = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        
        return


    def build_train_model(self):
        x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.maxseqlen])
        y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.maxseqlen])

        x_reverse = tf.reverse(x, [-1])

        x_list = []
        y_list = []
        for i in range(self.maxseqlen):
            x_list.append(x_reverse[:,i])
        for i in range(self.maxseqlen):
            y_list.append(y[:,i])

        outputs, states = embedding_attention_seq2seq(
                encoder_inputs = x_list,
                decoder_inputs = y_list,
                cell = self.LSTM,
                num_encoder_symbols = self.vocab_size,
                num_decoder_symbols = self.vocab_size,
                embedding_size = self.hidden_dim,
                output_projection = (tf.transpose(self.proj_w), self.proj_b)
        )

        # NCE loss: the target function to minimize
        nce_loss = 0.0
        for i in range(len(outputs)):
            loss = utils.nce_loss(outputs[i], y_list[i], self.proj_w, self.proj_b, 
                    self.vocab_size)
            loss = tf.reduce_sum(loss)

            nce_loss += loss

        nce_loss = nce_loss / tf.reduce_sum(tf.ones([self.batch_size, self.maxseqlen]))

        # Sigmoid cross entropy: the function for evaluation
        sigmoid_loss = 0.0
        for i in range(len(outputs)):
            loss = utils.sigmoid_loss(outputs[i], y_list[i], self.proj_w, self.proj_b, 
                    self.vocab_size)
            loss = tf.reduce_sum(loss)

            sigmoid_loss += loss

        sigmoid_loss = sigmoid_loss / tf.reduce_sum(tf.ones([self.batch_size, self.maxseqlen]))

        inputs = {
            'encoder_inputs': x,
            'decoder_inputs': y
        }

        loss = {
            'nce_loss': nce_loss,
            'sigmoid_loss': sigmoid_loss
        }

        return inputs, loss


    def build_test_model(self):
        x = tf.placeholder(dtype=tf.int32, shape=[1, self.maxseqlen])
        y = tf.placeholder(dtype=tf.int32, shape=[1, self.maxseqlen])

        x_reverse = tf.reverse(x, [-1])

        x_list = []
        y_list = []
        for i in range(self.maxseqlen):
            x_list.append(x_reverse[:,i])
        for i in range(self.maxseqlen):
            y_list.append(y[:,i])
        
        outputs, states = embedding_attention_seq2seq(
                encoder_inputs = x_list,
                decoder_inputs = y_list,
                cell = self.LSTM,
                num_encoder_symbols = self.vocab_size,
                num_decoder_symbols = self.vocab_size,
                embedding_size = self.hidden_dim,
                output_projection = (tf.transpose(self.proj_w), self.proj_b),
                feed_previous = True
        )

        caption = []

        for output in outputs:
            logits = tf.nn.xw_plus_b(output, tf.transpose(self.proj_w), self.proj_b)
            logits = tf.reshape(logits, [-1])
            logits = tf.gather(logits, np.arange(0, self.vocab_size-3))
            best_choice = tf.argmax(logits, axis=0)
            caption.append(best_choice)

        inputs = {
            'encoder_inputs': x,
            'decoder_inputs': y
        }

        return inputs, caption


    def save_model(self, sess, model_file):
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir(os.path.dirname(model_file))
        saver = tf.train.Saver()
        saver.save(sess, model_file)
        return


    def restore_model(self, sess, model_file):
        if os.path.isdir(os.path.dirname(model_file)):
            saver = tf.train.Saver()
            saver.restore(sess, model_file)
        return


