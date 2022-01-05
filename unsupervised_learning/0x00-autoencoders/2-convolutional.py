#!/usr/bin/env python3
"""Contains the function autoencoder()"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder

    Args:
        input_dims (int): contains dimensions of the model input
        filters (list): # of filters for each conv. layer
        latent_dims (int): int containing the dimensions of latent space
            representation

    Returns:
        encoder, decoder, auto
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """
    x = keras.layers.Input(shape=input_dims)
    encoded = x

    for i in range(len(filters)):
        encoded = keras.layers.Conv2D(
            filters[i], (3, 3), padding='same', activation='relu'
        )(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)

    decoded = keras.layers.Input(shape=latent_dims)
    x_decoded = decoded

    for i in range(len(filters) - 1, -1, -1):
        if i == 0:
            decoded = keras.layers.Conv2D(
                filters[i], (3, 3), padding='valid', activation='relu'
            )(decoded)
        else:
            decoded = keras.layers.Conv2D(
                filters[i], (3, 3), padding='same', activation='relu'
            )(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)

    decoded = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same'
    )(decoded)

    encoder = keras.Model(x, encoded)
    decoder = keras.Model(x_decoded, decoded)
    auto = keras.Model(x, decoder(encoder(x)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
