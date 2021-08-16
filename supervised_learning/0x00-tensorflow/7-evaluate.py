#!/usr/bin/env python3
"""Contains the function 'evaluate'
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluates the output of the neural network.

    Args:
      X: numpy.ndarray - contains input data to evaluate
      Y: numpy.ndarray - contains one-hot labels for X
      save_path: save location to load the model from

    Returns:
      the networks prediction, accuracy, and loss respectively
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        return sess.run([y_pred, loss, accuracy], feed_dict={x: X, y: Y})
