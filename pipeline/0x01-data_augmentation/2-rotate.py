#!/usr/bin/env python3
"""Contains the function rotate_image()"""

import tensorflow as tf


def rotate_image(image):
    """Rotates an image by 90 degrees counter-clockwise"""
    rotated = tf.image.rot90(image)

    return rotated
