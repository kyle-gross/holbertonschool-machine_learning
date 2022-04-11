#!/usr/bin/env python3
"""Contains the function flip_image()"""

import tensorflow  as tf


def flip_image(image):
    """Flips an image horizontally"""
    flipped = tf.image.flip_left_right(image)

    return flipped
