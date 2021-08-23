#!/usr/bin/env python3
"""Contains the functions shuffle_data(), forward_prop(), and model()
"""
import numpy as np
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Determines accuracy of neural network
    Args:
      y: placeholder for labels of input data
      y_pred: tensor containing network's predictions
    Returns:
      tensor containing the decimal accuracy of the prediction
    """
    y = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in Tensorflow
    using the Adam optimization algorithm.

    Args:
        loss: loss of network
        alpha: learning rate
        beta1: weight used for first moment
        beta2: weight used for second moment
        epsilon: small number to avoid division by 0

    Returns:
        Adam optimization operation
    """
    opt = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)

    return opt.minimize(loss)


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network in Tensorflow

    Args:
        prev: activated output of previous layer
        n: number of nodes in layer to be created
        actiavtion: activation function that should be used for the output
            of the layer

    Returns:
        Tensor of the activated output for the layer
    """
    # Create layer
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    base = tf.layers.Dense(n, kernel_initializer=kernel)

    # Init variables
    mean, variance = tf.nn.moments(base(prev), axes=[0])
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    epsilon = 1e-8

    batch_norm = tf.nn.batch_normalization(
        base(prev), mean, variance, beta, gamma, epsilon
    )

    return activation(batch_norm)


def create_placeholders(nx, classes):
    """
    Creates placeholders x and y.
    * @nx: number of feature columns in our data.
    * @classes: number of classes in our classifier
    Return: 2 placeholders
    """
    x = tf.placeholder(name='x', dtype=float, shape=[None, nx])
    y = tf.placeholder(name='y', dtype=float, shape=[None, classes])
    return x, y


def create_layer(prev, n, activation):
    """Creates a layer of neurons
    Args:
      prev: tensor output of the previous layer
      n: number of nodes in the layer to create
      activation: activation function that the layer should use
    Returns:
      tensor output of the layer
    """
    theta = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.dense(prev, n, activation, kernel_initializer=theta)


def forward_prop(prev, layers=[], activations=[]):
    """#all layers get batch_normalization but the last one,
    that stays without any activation or normalization"""
    y_pred = prev
    for i in range(len(layers)):
        if i < len(layers) - 1:
            y_pred = create_batch_norm_layer(y_pred, layers[i], activations[i])
        else:
            y_pred = create_layer(y_pred, layers[i], activations[i])

    return y_pred


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
    # Get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train, Y_train = Data_train[0], Data_train[1]
    X_valid, Y_valid = Data_valid[0], Data_valid[1]
    # Initialize x, y, y_pred, loss, accuracy
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations)
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    # Intialize alpha and global_step variable (not trainable)
    global_step = tf.Variable(0)
    alpha = tf.train.inverse_time_decay(
        alpha, global_step, 1, decay_rate, staircase=True
        )

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    # Create collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Get batch info
    m = X_train.shape[0]
    batches = m / batch_size
    if batches % 1 != 0:
        batches = int(batches) + 1
    else:
        batches = int(batches)

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
