#!/usr/bin/env python3
"""Contains the function autoencoder()"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates an autoencoder

    Args:
        input_dims (int): contains dimensions of the model input
        hidden_layers (list): # of nodes for each hidden layer
        latent_dims (int): int containing the dimensions of latent space
            representation

    Returns:
        encoder, decoder, auto
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """
    x = keras.layers.Input(shape=(input_dims,))
    x_encoded = keras.Input(shape=(latent_dims,))
    x_decoded = x_encoded

    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(x)

    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i], activation='relu'
        )(encoded)
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    
    decoded = keras.layers.Dense(
        hidden_layers[-1], activation='relu'
    )(x_decoded)

    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i], activation='relu'
        )(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    encoder = keras.models.Model(x, encoded)
    decoder = keras.models.Model(x_encoded, decoded)
    auto = keras.Model(x, decoder(encoder(x)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
