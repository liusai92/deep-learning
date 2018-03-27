"""
define models in this file.
"""

import tensorflow as tf


def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope("Convnet", reuse=reuse):

        conv1 = tf.layers.conv2d(x, filters=32, kernel_size=5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # flatten the data to 1-D vector
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, units=1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_classes)

        out = tf.nn.softmax(out) if not is_training else out

    return out
