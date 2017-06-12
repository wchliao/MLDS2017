import argparse
import time
import numpy as np
import tensorflow as tf
from DataSet import TrainData
from DataSet import TestData
from model import seq2seqModel


##### Global Constants #####

model_file = './model/model.ckpt'

############################


##### Parameters #####

batch_size = 256
maxseqlen = 22
embed_size = 256
num_layers = 4
init_learning_rate = 0.5
learning_rate_decay = 0.9
min_learning_rate = 0.0001
max_epoch = 10
display_step = 10

######################


##### GPU Options #####

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

#######################


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--test', action='store_true', help='Run testing')
    parser.add_argument('-t', '--datafile', help='.npy data file')
    parser.add_argument('-d', '--dictionary', help='dictionary')
    parser.add_argument('-q', '--question', help='testing file here')
    parser.add_argument('-o', '--output', help='output file here')
    
    return parser.parse_args()


def vec2line(mapped_sent, dictionary):
    sent = ''
    for i, word in enumerate(mapped_sent):
        if dictionary[word] == '<EOS>':
            break
        if i > 0:
            sent += ' '
        sent += dictionary[word]

    return sent


def train(datafile, dictfile):
    data = TrainData(datafile, dictfile, maxseqlen = maxseqlen)

    model = seq2seqModel(
            batch_size = batch_size,
            vocab_size = data.dictsize,
            maxseqlen = maxseqlen,
            embed_size = embed_size,
            num_layers = num_layers
    )
    inputs, loss, optimize = model.build_train_model()

    init = tf.global_variables_initializer()
    learning_rate = init_learning_rate

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        model.restore_model(sess, model_file)

        step = 0
        start_time = time.time()

        while data.epoch < max_epoch:
            x, y, y_seqlen = data.next_batch(batch_size = batch_size)

            sess.run(optimize, feed_dict={
                inputs['encoder_inputs']: x,
                inputs['decoder_inputs']: y,
                inputs['decoder_length']: y_seqlen,
                inputs['learning_rate']: learning_rate
            })

            if step % display_step == 0:
                cur_loss = sess.run(loss['softmax_loss'], feed_dict={
                    inputs['encoder_inputs']: x,
                    inputs['decoder_inputs']: y,
                    inputs['decoder_length']: y_seqlen
                })
                
                model.save_model(sess, model_file)
                
                used_time = time.time() - start_time
                
                print('{} / {} steps: softmax loss = {} time = {} secs'.format(
                    step, max_epoch * data.datasize // batch_size, cur_loss, used_time)
                )

                learning_rate = init_learning_rate * pow(learning_rate_decay, data.epoch)
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

                start_time = time.time()
            
            step += 1
        
    return


def test(questionfile, dictfile):
    data = TestData(questionfile, dictfile, maxseqlen = maxseqlen)
    inv_dictionary = {value: key for (key, value) in data.dict.items()}

    model = seq2seqModel(
            batch_size = 1,
            vocab_size = data.dictsize,
            maxseqlen = maxseqlen,
            embed_size = embed_size,
            num_layers = num_layers
    )
    inputs, caption = model.build_test_model()

    init = tf.global_variables_initializer()

    sents = []

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        model.restore_model(sess, model_file)

        for _ in range(data.datasize):
            x, y = data.next_batch()

            mapped_sent = sess.run(caption, feed_dict={
                inputs['encoder_inputs']: x,
                inputs['decoder_inputs']: y
            })

            sent = vec2line(mapped_sent, inv_dictionary)
            sents.append(sent)

    return sents


def WriteResults(results, outputfile):
    with open(outputfile, 'w') as f:
        for caption in results:
            f.write(caption)
            f.write('\n')
    return


if __name__ == '__main__':
    args = parse_args()
    if args.train:
        train(args.datafile, args.dictionary)
    if args.test:
        results = test(args.question, args.dictionary)
        WriteResults(results, args.output)

