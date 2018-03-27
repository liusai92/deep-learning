#!usr/bin/env python3
"""
main file, training the model in this file.
"""

import tensorflow as tf
from dataset import get_imagepaths_and_labels, read_images
from dataset import get_test_imagepaths_labels, read_test_images
from models import conv_net


# paths setting
MODEL_PATH = './models/conv_net'
DATASET_PATH = ''


# training parameters
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 100

# network parameters
num_classes = 10
dropout = 0.75


# reading the dataset
imagepaths, labels = get_imagepaths_and_labels(DATASET_PATH)
imagepaths_test, labels_test = get_test_imagepaths_labels(DATASET_PATH)

# build data input
X, Y = read_images(imagepaths, labels, batch_size)
X_test, Y_test = read_test_images(imagepaths_test, labels_test)

# define training ops
logits_train = conv_net(X, num_classes, dropout, reuse=False, is_training=True)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits \
                                           (logits=logits_train, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# define testing ops
logits_test = conv_net(X_test, num_classes, dropout, reuse=True, is_training=False)
correct_pred = tf.equal(tf.argmax(logits_test, 1), Y_test)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

# training
with tf.Session() as sess:
    
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print("Start training")

    for step in range(1, num_steps+1):
        
        if step % display_step == 0:
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step: " + "%04d" %step + ", loss= " + "{:.9f}".format(loss)\
                   + ", accuracy:" + "{:.4f}".format(acc))
        else:
            sess.run(train_op)

    print("Optimization finished!")

    saver.save(sess, MODEL_PATH)

    coord.request_stop()
    coord.join(threads)
