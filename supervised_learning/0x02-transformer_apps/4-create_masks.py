#!/usr/bin/env python3
"""Contains the function create_masks()"""

import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """Creates all masks for training/validation
    Args:
        inputs [tf.Tensor] (batch_size, seq_len_in):
            contains input sentence
        target [tf.Tensor] (batch_size, seq_len_out):
            contains target sentence
    Returns:
        encoder_mask, combined_mask, decoder_mask

        encoder_mask [tf.Tensor] (batch_size, 1, 1, seq_len_in):
                to be applied in the encoder
        combined_mask [tf.Tensor] (batch_size, 1, seq_len_out, seq_len_out
            used in the 1st attention block in the decoder to pad and mask
            future tokens in the input received by the decoder
        decoder_mask [tf.Tensor] (batch_size, 1, 1, seq_len_in):
            used in the 2nd attention block in the decoder
    """
    batch_size, seq_len_out = target.shape

    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0
    )
    look_ahead_mask = 1 - look_ahead_mask

    padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

    combined_mask = tf.maximum(look_ahead_mask, padding_mask)

    return encoder_mask, combined_mask, decoder_mask
