#!/usr/bin/env python3
"""Contains the function autoencoder()"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder

    Args:
        input_dims (int): contains dimensions of the model input
        hidden_layers (list): # of nodes for each hidden layer
        latent_dims (int): int containing the dimensions of latent space
            representation
        lambtha (float): regularization parameter used for L1 regularization

    Returns:
        encoder, decoder, auto
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """
    x = keras.layers.Input(shape=(input_dims,))
    L1 = keras.regularizers.l1(lambtha)

    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(x)

    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i], activation='relu'
        )(encoded)
    encoded = keras.layers.Dense(
        latent_dims, activation='relu', activity_regularizer=L1
    )(encoded)

    decoded = keras.layers.Input(shape=(latent_dims,))
    x_decoded = decoded

    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i], activation='relu'
        )(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    encoder = keras.Model(x, encoded)
    decoder = keras.Model(x_decoded, decoded)
    auto = keras.Model(x, decoder(encoder(x)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
