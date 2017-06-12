import numpy as np
import tensorflow as tf
import utils
import os


class seq2seqModel(object):
    def __init__(self, batch_size, vocab_size, 
            maxseqlen=20, embed_size=256, num_layers=1):

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.maxseqlen = maxseqlen

        self.proj_w = tf.get_variable('proj_w', [embed_size, vocab_size])
        self.proj_b = tf.get_variable('proj_b', [vocab_size])

        single_cell = tf.contrib.rnn.BasicLSTMCell(embed_size)
        if num_layers > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)
        else:
            self.cell = single_cell
        
        return


    def build_train_model(self):
        x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.maxseqlen])
        y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.maxseqlen])
        y_seqlen = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        learning_rate = tf.placeholder(dtype=tf.float32)

        x_reverse = tf.reverse(x, [-1])

        x_list = [x_reverse[:,i] for i in range(self.maxseqlen)]
        y_list = [y[:,i] for i in range(self.maxseqlen)]

        targets_list = [y_list[i+1] for i in range(len(y_list)-1)]
        targets_list.append(y_list[-1])  # Just use to fit the shape

        targets_mask = tf.sequence_mask(y_seqlen, self.maxseqlen, dtype=tf.float32)
        targets_mask_list = [targets_mask[:,i] for i in range(len(targets_list))]

        outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs = x_list,
                decoder_inputs = y_list,
                cell = self.cell,
                num_encoder_symbols = self.vocab_size,
                num_decoder_symbols = self.vocab_size,
                embedding_size = self.embed_size,
                output_projection = (self.proj_w, self.proj_b)
        )

        # Sampled softmax loss: the target function to minimize
        sampled_loss = tf.contrib.legacy_seq2seq.sequence_loss(
                logits = outputs,
                targets = targets_list,
                weights = targets_mask_list,
                softmax_loss_function = \
                    lambda y, x: \
                    utils.sampled_loss(y, x, tf.transpose(self.proj_w), self.proj_b, self.vocab_size)
        )

        # Softmax loss: the function for evaluation
        softmax_loss = tf.contrib.legacy_seq2seq.sequence_loss(
                logits = outputs,
                targets = targets_list,
                weights = targets_mask_list,
                softmax_loss_function = \
                    lambda y, x: \
                    utils.softmax_loss(y, x, self.proj_w, self.proj_b, self.vocab_size)
        )

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(sampled_loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) \
                for (grad, var) in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

        inputs = {
            'encoder_inputs': x,
            'decoder_inputs': y,
            'decoder_length': y_seqlen,
            'learning_rate': learning_rate
        }

        loss = {
            'sampled_loss': sampled_loss,
            'softmax_loss': softmax_loss
        }

        return inputs, loss, train_op


    def build_test_model(self):
        x = tf.placeholder(dtype=tf.int32, shape=[1, self.maxseqlen])
        y = tf.placeholder(dtype=tf.int32, shape=[1, self.maxseqlen])

        x_reverse = tf.reverse(x, [-1])

        x_list = [x_reverse[:,i] for i in range(self.maxseqlen)]
        y_list = [y[:,i] for i in range(self.maxseqlen)]
        
        outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs = x_list,
                decoder_inputs = y_list,
                cell = self.cell,
                num_encoder_symbols = self.vocab_size,
                num_decoder_symbols = self.vocab_size,
                embedding_size = self.embed_size,
                output_projection = (self.proj_w, self.proj_b),
                feed_previous = True
        )

        caption = []

        for output in outputs:
            logits = tf.nn.xw_plus_b(output, self.proj_w, self.proj_b)
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

