"""
create models in this file.
"""

import tensorflow as tf

def lenet(x):
    """ creat lenet. """
    with tf.variable_scope('lenet') as scope:
        # data:[-1, 28, 28, 1]
        conv_1 = tf.layers.conv2d(x, filters=6, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
        # data:[-1, 28, 28, 1]
        conv_1 = tf.layers.max_pooling2d(conv_1, [2,2], strides=2)
        # data:[-1, 14, 14, 1]
        conv_2 = tf.layers.conv2d(conv_1, 16, [5,5], padding='valid', activation=tf.nn.relu)
        # data:[-1, 10, 10, 1]
        conv_2 = tf.layers.max_pooling2d(conv_2, [2,2], strides=2)
        # data:[-1, 5, 5, 1]
        conv_3 = tf.layers.conv2d(conv_2, 120, [5,5], padding='valid', activation=tf.nn.relu)
        # data:[-1, 1, 1, 1]
        fl = tf.layers.flatten(conv_3)
        # data:[-1, 1]
        fc1 = tf.layers.dense(fl, units=84, activation=tf.nn.relu)

        out = tf.layers.dense(fc1, units=10, activation=tf.nn.tanh)

    return out
