#!/usr/bin/python3
"""
The main file for training the model.
"""

import tensorflow as tf
import os
from model import lenet
from dataset import dataloader_train


# parameters
TRAIN_MODEL_PATH = './models/train/lenet'
PRED_MODEL_PATH = './models/predict/lenet'
LOG_PATH = './log'
TRAIN_DATA_PATH = './data/train.csv'

num_steps = 1000
display_step = 100

batch_size = 128

start_learning_rate = 0.01
decay_rate = 0.2

# construct input
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
Y = tf.placeholder(dtype=tf.float32, shape=(None, 10))

# logit
logits = lenet(X)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
tf.summary.scalar('loss', loss_op)

# optimizer
learning_rate = tf.Variable(start_learning_rate, trainable=False)
tf.summary.scalar('learning_rate', learning_rate)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

# sumary & saver
summary = tf.summary.merge_all()
saver_train = tf.train.Saver()
sav_pred_variables = {}
for v in tf.get_default_graph().get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    if v.name.startswith('lenet') and ('Adam' not in v.name):
        sav_pred_variables[v.name.strip(':0')] = v
saver_pred = tf.train.Saver(sav_pred_variables)
# start training

with tf.Session() as sess:

    print("Start training!")

    if os.path.exists(TRAIN_MODEL_PATH + '.meta'):
        saver_train.restore(sess, TRAIN_MODEL_PATH)
    else:
        sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

    with dataloader_train(batch_size, TRAIN_DATA_PATH) as dataloader:

        for step in range(1, num_steps+1):
            sess.run(tf.assign(learning_rate, start_learning_rate * \
                    decay_rate ** (step/num_steps)))

            images, labels = dataloader.next_batch()

            if step % display_step == 0:

                summ, loss, _ = sess.run([summary, loss_op, train_op], feed_dict={X: images, Y: labels})
                print("Step:" + "%04d" %step + ", loss=" + "{:.9f}".format(loss))

            else:
                summ, _ = sess.run([summary, train_op], feed_dict={X: images, Y: labels})

            writer.add_summary(summ, global_step=step)
        print("Optimization finished!")
        writer.close()
        saver_train.save(sess, TRAIN_MODEL_PATH)
        saver_pred.save(sess, PRED_MODEL_PATH, write_meta_graph=False)
