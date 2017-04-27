log_beam_probs, beam_symbols, beam_path  = [], [], []
def beam_search(prev, i):
    probs = tf.log(tf.nn.softmax(prev))
    if i > 1:
        probs = tf.reshape(probs + log_beam_probs[-1], 
                           [-1, beam_size * num_symbols])

    best_probs, indices = tf.nn.top_k(probs, beam_size)
    indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
    best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

    symbols = indices % num_symbols # Which word in vocabulary.
    beam_parent = indices // num_symbols # Which hypothesis it came from.

    beam_symbols.append(symbols)
    beam_path.append(beam_parent)
    log_beam_probs.append(best_probs)
