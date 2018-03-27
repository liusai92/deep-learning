"""
generate train or test data tensor in this file.
"""

import tensorflow as tf
import os
import numpy as np
from scipy.misc import imread

# image parameters

CHANNELS = 3
IMG_HEIGHT = 64
IMG_WIDTH = 64


# walk through the data folder

def get_imagepaths_and_labels(dataset_path):
    
    train_path = os.path.join(dataset_path, 'train')
    imagepaths, labels = [], []
    label_id = get_label_id()   

    for label in os.listdir(train_path):
        for fnm in os.listdir(os.path.join(train_path, label, 'images')):
            imagepaths.append(os.path.join(train_path, label, 'images', fnm))
            labels.append(label_id[label])
    return imagepaths, labels


# read images from disk and set tf queue

def read_images(imagepaths, labels, batch_size):
    # convert to tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # build a tf queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) * tf.constant(1.0) / tf.constant(127.5) \
                                                          - tf.constant(1.0)
    X, Y = tf.train.batch([image, label], batch_size=batch_size,\
                          capacity=batch_size * 8,\
                          num_threads=4)
    return X, Y


# get val_annotations filename --> label

def get_val_annotations(dataset_path):
    val_path = os.path.join(dataset_path, 'val')
    val_anno = {}
    with open(os.path.join(val_path, 'val_annotations.txt')) as f:
        for line in f.readlines():
            tmp = line.split('\t')
            # tmp[0] val_*.JPEG tmp[1] n0***
            val_anno[tmp[0]] = tmp[1]
    return val_anno

# get label --> id

def get_label_id():
    label_id = {}
    with open('./10ids.txt') as f:
        for line in f.readlines():
            [label, id] = line.strip().split()
            label_id[label] = int(id)
    return label_id

def get_test_imagepaths_labels(dataset_path):
    
    test_path = os.path.join(dataset_path, 'val', 'images')

    val_anno = get_val_annotations(dataset_path)
    label_id = get_label_id()

    imagepaths, labels = [], []

    for fname in os.listdir(test_path):
        imagepaths.append(os.path.join(test_path, fname))
        lb = val_anno[fname]
        labels.append(label_id[lb])

    return imagepaths, labels


def read_test_images(imagepaths, labels):
    
    num_samples = len(imagepaths)
    images = np.empty(shape=(num_samples, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    
    for i in range(num_samples):
        images[i,:] = np.array(imread(imagepaths[i], mode='RGB'))

    # normalization
    images = images/127.5 - 1.0

    # convert to tf.Tensor
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)

    return images, labels
