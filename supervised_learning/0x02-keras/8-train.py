#!/usr/bin/env python3
"""Contains the function train_model()"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """Trains a Keras model using mini-batch gradient descent

    Args:
        network (keras model): model to train
        data (numpy.ndarray)(m, nx): input data
        labels (numpy.ndarray)(m, classes): contains labels of data
        batch_size (int): size of batch for gradient descent
        epochs (int): number of passes through training set
        validation_data (numpy.ndarray): validation set
        early_stopping (bool): if True, determines stopping point
        patience (int): patience used for early stopping
        learning_rate_decay (bool): if True, uses learning rate decay
        alpha (float): initial learning rate
        decay_rate (int): rate of decay
        save_best (bool): if True, save best model
        filepath (str): path to where the model should be saved
        verbose (bool): if True print output
        shuffle (bool): if True, shuffle data between epochs

    Returns:
        generated History object
    """
    if validation_data:
        callback = list()
        if early_stopping:
            callback.append(K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            ))
        if learning_rate_decay:
            def schedule(epoch):
                """Returns new learning rate (float)"""
                return alpha / (1 + epoch * decay_rate)

            callback.append(K.callbacks.LearningRateScheduler(
                schedule,
                verbose=1
            ))
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callback,
        validation_data=validation_data,
        shuffle=shuffle
    )
    if save_best:
        network.save(filepath)

    return history
