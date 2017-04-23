from __future__ import division
import tensorflow as tf
import numpy as np

with tf.Graph().as_default():
    beam_size = 3 # Number of hypotheses in beam.
    num_symbols = 5 # Output vocabulary size.
    embedding_size = 10
    num_steps = 3
    embedding = tf.zeros([num_symbols, embedding_size])
    output_projection = None

    # log_beam_probs: list of [beam_size, 1] Tensors
    #  Ordered log probabilities of the `beam_size` best hypotheses
    #  found in each beam step (highest probability first).
    # beam_symbols: list of [beam_size] Tensors 
    #  The ordered `beam_size` words / symbols extracted by the beam
    #  step, which will be appended to their corresponding hypotheses
    #  (corresponding hypotheses found in `beam_path`).
    # beam_path: list of [beam_size] Tensor
    #  The ordered `beam_size` parent indices. Their values range
    #  from [0, `beam_size`), and they denote which previous
    #  hypothesis each word should be appended to.
    log_beam_probs, beam_symbols, beam_path  = [], [], []
    def beam_search(prev, i):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])

        # Compute 
        #  log P(next_word, hypothesis) = 
        #  log P(next_word | hypothesis)*P(hypothesis) =
        #  log P(next_word | hypothesis) + log P(hypothesis)
        # for each hypothesis separately, then join them together 
        # on the same tensor dimension to form the example's 
        # beam probability distribution:
        # [P(word1, hypothesis1), P(word2, hypothesis1), ...,
        #  P(word1, hypothesis2), P(word2, hypothesis2), ...]

        # If TF had a log_sum_exp operator, then it would be 
        # more numerically stable to use: 
        #   probs = prev - tf.log_sum_exp(prev, reduction_dims=[1])
        probs = tf.log(tf.nn.softmax(prev))
        # i == 1 corresponds to the input being "<GO>", with
        # uniform prior probability and only the empty hypothesis
        # (each row is a separate example).
        if i > 1:
            probs = tf.reshape(probs + log_beam_probs[-1], 
                               [-1, beam_size * num_symbols])

        # Get the top `beam_size` candidates and reshape them such
        # that the number of rows = batch_size * beam_size, which
        # allows us to process each hypothesis independently.
        best_probs, indices = tf.nn.top_k(probs, beam_size)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

        symbols = indices % num_symbols # Which word in vocabulary.
        beam_parent = indices // num_symbols # Which hypothesis it came from.

        beam_symbols.append(symbols)
        beam_path.append(beam_parent)
        log_beam_probs.append(best_probs)
        return tf.nn.embedding_lookup(embedding, symbols)

    # Setting up graph.
    inputs = [tf.placeholder(tf.float32, shape=[None, num_symbols])
              for i in range(num_steps)]
    for i in range(num_steps):
        beam_search(inputs[i], i + 1)

    # Running the graph.
    input_vals = [0, 0, 0]
    l = np.log
    eps = -10 # exp(-10) ~= 0

    # These values mimic the distribution of vocabulary words
    # from each hypothesis independently (in log scale since
    # they will be put through exp() in softmax).
    input_vals[0] = np.array([[0, eps, l(2), eps, l(3)]])
    # Step 1 beam hypotheses =
    # (1) Path: [4], prob = log(1 / 2)
    # (2) Path: [2], prob = log(1 / 3)
    # (3) Path: [0], prob = log(1 / 6)

    input_vals[1] = np.array([[l(1.2), 0, 0, l(1.1), 0], # Path [4] 
                              [0,   eps, eps, eps, eps], # Path [2]
                              [0,  0,   0,   0,   0]])   # Path [0]
    # Step 2 beam hypotheses =
    # (1) Path: [2, 0], prob = log(1 / 3) + log(1)
    # (2) Path: [4, 0], prob = log(1 / 2) + log(1.2 / 5.3)
    # (3) Path: [4, 3], prob = log(1 / 2) + log(1.1 / 5.3)

    input_vals[2] = np.array([[0,  l(1.1), 0,   0,   0], # Path [2, 0]
                              [eps, 0,   eps, eps, eps], # Path [4, 0]
                              [eps, eps, eps, eps, 0]])  # Path [4, 3]
    # Step 3 beam hypotheses =
    # (1) Path: [4, 0, 1], prob = log(1 / 2) + log(1.2 / 5.3) + log(1)
    # (2) Path: [4, 3, 4], prob = log(1 / 2) + log(1.1 / 5.3) + log(1)
    # (3) Path: [2, 0, 1], prob = log(1 / 3) + log(1) + log(1.1 / 5.1)

    input_feed = {inputs[i]: input_vals[i][:beam_size, :] 
                  for i in range(num_steps)} 
    output_feed = beam_symbols + beam_path + log_beam_probs
    session = tf.InteractiveSession()
    outputs = session.run(output_feed, feed_dict=input_feed)

    expected_beam_symbols = [[4, 2, 0],
                             [0, 0, 3],
                             [1, 4, 1]]
    expected_beam_path = [[0, 0, 0],
                          [1, 0, 0],
                          [1, 2, 0]]

    print("predicted beam_symbols vs. expected beam_symbols")
    for ind, predicted in enumerate(outputs[:num_steps]):
        print(list(predicted), expected_beam_symbols[ind])
    print("\npredicted beam_path vs. expected beam_path")
    for ind, predicted in enumerate(outputs[num_steps:num_steps * 2]):
        print(list(predicted), expected_beam_path[ind])
    print("\nlog beam probs")
    for log_probs in outputs[2 * num_steps:]:
        print(log_probs)