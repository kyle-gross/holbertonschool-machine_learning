#!/usr/bin/env python3
"""Contains the function crop_image()"""

import tensorflow as tf


def crop_image(image, size):
    """Performs a random crop of an image"""
    cropped = tf.image.random_crop(image, size)

    return cropped
