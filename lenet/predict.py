#!/usr/bin/python3
"""
The main file for predicting.
"""

import tensorflow as tf
from model import lenet
from dataset import dataloader_predict

# parameters
PRED_MODEL_PATH = './models/predict/lenet'
PRED_DATA_PATH = './data/test.csv'

# create input
X = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])

# logits
logits = lenet(X)

# prediction
pred = tf.argmax(tf.nn.softmax(logits=logits), axis=1)

saver = tf.train.Saver()

with tf.Session() as sess:
    
    saver.restore(sess, PRED_MODEL_PATH)
    
    with dataloader_predict(PRED_DATA_PATH) as dataloader:
        imageid = 1
        while True:
            try:
                label = sess.run(pred, feed_dict={X: dataloader.next_line()})
                dataloader.writer.writerow([imageid, label[0]])
                imageid += 1
            except StopIteration:
                break

    print("Prediction finished!")
