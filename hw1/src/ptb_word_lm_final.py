"""
To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "test",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_string("q_path", None,
                  "Question Path")
flags.DEFINE_string("p_path", None,
                  "prediction Path")


FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)

class QuestionInput(object):
  def __init__(self, questions):
    self.batch_size = len(questions)
    self.input_data = questions

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    if is_training:
      batch_size = input_.batch_size
      num_steps = input_.num_steps
    else:
      batch_size = config.batch_size
      num_steps = config.num_steps

    size = config.hidden_size
    vocab_size = config.vocab_size

    self.question = tf.placeholder(tf.int32, [1, None])

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      if is_training:
        inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
      else:
        inputs = tf.nn.embedding_lookup(embedding, self.question)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    
    self._logits = tf.nn.softmax(tf.matmul(output, softmax_w) + softmax_b)

    #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    #    [logits],
    #    [tf.reshape(input_.targets, [-1])],
    #    [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._final_state = state


    if not is_training:
      return

    num_samples = 10
    labels = tf.reshape(input_.targets, [-1,1])
    hidden = output
    w_t = tf.transpose(softmax_w)
    loss = tf.nn.sampled_softmax_loss(w_t, softmax_b,
                                      labels,
                                      hidden,  
                                      num_samples,
                                      vocab_size)

    self._cost = cost = tf.reduce_sum(loss) / batch_size

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def logits(self):
    return self._logits

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 30
  hidden_size = 200
  max_epoch = 2
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 30
  vocab_size = 12001


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 30
  hidden_size = 200
  max_epoch = 1
  max_max_epoch = 0
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  vocab_size = 12001


def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def predict(session, model, questions):
  """Runs the model on the given data."""
  with open(FLAGS.p_path, "w") as f:
    q_id = 1
    f.write("id,answer\n")
    for q in questions:
      left = q.left
      if len(left) < 30:
        for i in range(30-len(left)):
          left += [0]

      start_time = time.time()
      costs = 0.0
      iters = 0
      state = session.run(model.initial_state)

      fetches = {
          "logits": model.logits,
          "final_state": model.final_state,
      }

      feed_dict = {}
      for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      feed_dict[model.question] = np.array(left).reshape((1,-1))

      vals = session.run(fetches, feed_dict)
     
      p_opt = []

      for o in q.options:
        p_opt.append(np.log(vals["logits"][q.pos-1, o]))
      f.write("{},{}\n".format(q_id, ["a", "b", "c", "d", "e"][np.argmax(p_opt)]))
      q_id += 1

      #logits = session.run(model.logits)
      #print(logits[0])

  return None


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  raw_data = reader.load_holmes_data(12001)
  train_data, _ , word_to_id = raw_data
  test_questions = reader.get_questions(word_to_id)#, FLAGS.q_path)

  
  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 30

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=train_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config,
                         input_=test_questions)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    print(FLAGS.save_path)
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.05
    with sv.managed_session(config=session_config) as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)

      test_perplexity = predict(session, mtest, test_questions)

      #if FLAGS.save_path:
      #  print("Saving model to %s." % FLAGS.save_path)
      #  sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()
