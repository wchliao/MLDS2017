import tensorflow as tf


def nce_loss(x, y, w, b, vocab_size, num_samples=64):
    y = tf.reshape(y, [-1, 1])
    
    return tf.nn.nce_loss(
            weights = w,
            biases = b,
            labels = y,
            inputs = x,
            num_sampled = num_samples,
            num_classes = vocab_size
            )


def sigmoid_loss(x, y, w, b, vocab_size):
    logits = tf.matmul(x, tf.transpose(w))
    logits = tf.nn.bias_add(logits, b)
    y_one_hot = tf.one_hot(y, vocab_size)
    
    return tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_one_hot,
            logits=logits
            )
