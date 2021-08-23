#!/usr/bin/env python3
"""Contains the functions shuffle_data(), forward_prop(), and model()
"""
import numpy as np
import tensorflow as tf


def forward_prop(prev, layers=[], activations=[], epsilon=1e-8):
    """#all layers get batch_normalization but the last one,
    that stays without any activation or normalization"""
    for i, n in enumerate(layers):
        kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        dense = tf.layers.Dense(n, kernel_initializer=kernel, name='dense')
        z = dense(prev)
        if i < len(layers) - 1:
            mean, variance = tf.nn.moments(z, axes=[0])
            gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
            beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
            z_norm = tf.nn.batch_normalization(
                z, mean, variance, beta, gamma, epsilon
            )
            prev = activations[i](z_norm)
        else:
            prev = z

    return prev


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices

    Args:
        X: numpy.ndarray - shape: (m, nx)
          * m = number of data points
          * nx = number of features in X
        Y: numpy.ndarray - shape: (m, ny)

    Returns:
        shuffled X and Y matrices
    """
    shuffle = np.random.permutation(X.shape[0])

    return X[shuffle], Y[shuffle]


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """Builds, trains, and saves a neural network model in tensorflow
    using Adam optimization, mini-batch gradient descent, learning rate
    decay, and batch normalization.

    Args:
        Data_train: tuple - contains training inputs/labels
        Data_valid: tuple - contains validation inputs/labels
        layers: list - contains number of nodes in each layer of network
        activations: list - contains the activation functions used for each
            layer
        alpha: leanring rate
        beta1: weight for first moment of Adam opt.
        beta2: weight for second moment of Adam opt.
        epsilon: small number used to avoid division by zero
        decay_rate: decay rate for inverse time decay
            * Corresponding decay step should be 1
        batch_size: number of data points per mini batch
        epochs: number of times to pass through dataset
        save_path: path to where the model will be saved

    Returns:
        Path to model
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid

    m, nx = X_train.shape
    classes = Y_train.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    tf.add_to_collection('loss', loss)

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    # batches == decay_steps
    batches = m // batch_size
    if batches % batch_size != 0:
        batches += 1

    alpha = tf.train.inverse_time_decay(
        alpha, global_step, batches, decay_rate, staircase=True
        )
    train_op = tf.train.AdamOptimizer(
        alpha, beta1, beta2, epsilon
    ).minimize(loss, global_step)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

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
        saver = tf.train.Saver()

        return saver.save(sess, save_path)
