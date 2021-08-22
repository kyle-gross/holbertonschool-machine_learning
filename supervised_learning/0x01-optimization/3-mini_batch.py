#!/usr/bin/env python3
"""Contains the function train_mini_batch.
"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """Trains a loaded neural network model using mini-batch
    gradient descent.

    Args:
        X_train: numpy.ndarray (m, 784) - Contains training data
            * m: number of data points
            * 784: number of input features
        Y_train: one-hot numpy.ndarray (m, 10) - Contains training labels
            * 10: number of classes
        X_valid: numpy.ndarray (m, 784) - Contains validation data
        Y_valid: numpy.ndarray (m, 10) - Contains validation labels
        batch_size: number of data points in a batch
        epochs: number of times the training should pass through the whole
            dataset
        load_path: path from which to load the model
            Loaded model collection:
            * x: placeholder for input data
            * y: placeholder for the labels
            * accuracy: op to calculate the accuracy of the model
            * loss: op to calculate the cost of the model
            * train_op: op to perform one pass of gradient descent
        save_path: path to where the model should be saved after training

    Returns:
        Path to model
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)

        m = X_train.shape[0]
        batches = int(m / batch_size)
        if batches % 1 != 0:
            batches += 1

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs + 1):
            train_cost = loss.eval({x: X_train, y: Y_train})
            train_accuracy = accuracy.eval({x: X_train, y: Y_train})
            valid_cost = loss.eval({x: X_valid, y: Y_valid})
            valid_accuracy = accuracy.eval({x: X_valid, y: Y_valid})

            print('After {} epochs:\n'.format(i) +
                  '\tTraining Cost: {}\n'.format(train_cost) +
                  '\tTraining Accuracy: {}\n'.format(train_accuracy) +
                  '\tValidation Cost: {}\n'.format(valid_cost) +
                  '\tValidation Accuracy: {}'.format(valid_accuracy))

            if i < epochs:
                X_shuffle, Y_shuffle = shuffle_data(X_train, Y_train)
                for j in range(batches):
                    feed_dict = {
                        x: X_shuffle[batch_size * j:batch_size * (j+1)],
                        y: Y_shuffle[batch_size * j:batch_size * (j+1)]
                    }

                    sess.run(train_op, feed_dict)

                    if (j + 1) % 100 == 0 and j != 0:
                        step_cost = loss.eval(feed_dict)
                        step_accuracy = accuracy.eval(feed_dict)
                        print('\tStep {}:\n'.format(j + 1) +
                              '\t\tCost: {}\n'.format(step_cost) +
                              '\t\tAccuracy: {}'.format(step_accuracy))

        return saver.save(sess, save_path)
