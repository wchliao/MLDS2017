#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf 

sess = tf.Session()
params = tf.constant([6, 3, 4, 1, 5, 9, 10])
indices = tf.constant([2, 0, 2, 5])
output = tf.gather(params, indices)
print(sess.run(output))
sess.close()