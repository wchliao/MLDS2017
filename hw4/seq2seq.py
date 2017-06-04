import argparse
import time
import numpy as np
import tensorflow as tf
from DataSet import TrainData
from DataSet import TestData
from model import seq2seqModel
import DataPreprocessor


##### Global Constants #####

model_file = './model/model.ckpt'

############################


##### Parameters #####

batch_size = 256
maxseqlen = 30
hidden_dim = 256
learning_rate = 0.5
max_epoch = 20
display_step = 1000

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


def train(datafile, dictfile):
    data = TrainData(datafile, dictfile, maxseqlen = maxseqlen)

    model = seq2seqModel(
            batch_size = batch_size,
            vocab_size = data.dictsize,
            maxseqlen = maxseqlen,
            hidden_dim = hidden_dim,
    )
    inputs, loss = model.build_train_model()

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss['nce_loss'])

    init = tf.global_variables_initializer()

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        model.restore_model(sess, model_file)

        step = 0
        start_time = time.time()

        while data.epoch < max_epoch:
            x, y = data.next_batch(batch_size = batch_size)

            sess.run(optimizer, feed_dict={
                inputs['encoder_inputs']: x,
                inputs['decoder_inputs']: y
            })

            if step % display_step == 0:
                epoch = data.epoch
                model.save_model(sess, model_file)
                used_time = time.time() - start_time

                cur_loss = sess.run(loss['sigmoid_loss'], feed_dict={
                    inputs['encoder_inputs']: x,
                    inputs['decoder_inputs']: y
                })

                print(str(step) + '/' + str(max_epoch * data.datasize // batch_size) + ' steps: ' +
                        'loss = ' + str(np.mean(cur_loss)) + ' ' + 
                        'time = ' + str(used_time) + ' secs'
                )

                start_time = time.time()
            
            step += 1
        
        model.save_model(sess, model_file)

    return


def test(inputfile, dictfile):
    data = TestData(inputfile, dictfile, maxseqlen = maxseqlen)
    inv_dictionary = {value: key for (key, value) in data.dict.items()}

    model = seq2seqModel(
            batch_size = batch_size,
            vocab_size = data.dictsize,
            maxseqlen = maxseqlen,
            hidden_dim = hidden_dim,
    )
    inputs, output = model.build_test_model()

    init = tf.global_variables_initializer()
    
    results = []

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        model.restore_model(sess, model_file)

        while data.index < data.datasize:
            x, y = data.next_batch()

            mapped_output = sess.run(output, feed_dict={
                inputs['encoder_inputs']: x,
                inputs['decoder_inputs']: y
            })

            caption = ''

            for i, word in enumerate(mapped_output):
                if inv_dictionary[word] == data.dict['<EOS>']:
                    break
                else:
                    if i > 0:
                        caption += ' '
                    caption += inv_dictionary[word]

            results.append(caption)

    return results


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

