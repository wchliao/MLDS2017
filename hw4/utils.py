import tensorflow as tf


def sampled_loss(y, x, w, b, vocab_size, num_samples=512):
    return tf.nn.sampled_softmax_loss(
            weights = w,
            biases = b,
            labels = tf.expand_dims(y, -1),
            inputs = x,
            num_sampled = num_samples,
            num_classes = vocab_size
            )


def softmax_loss(y, x, w, b, vocab_size):
    logits = tf.matmul(x, w)
    logits = tf.nn.bias_add(logits, b)
    y_one_hot = tf.one_hot(y, vocab_size)

    return tf.nn.softmax_cross_entropy_with_logits(
            labels = y_one_hot,
            logits = logits
            )
    
