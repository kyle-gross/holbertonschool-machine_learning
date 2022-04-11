#!/usr/bin/env python3
"""Contains the function shear_image()"""

from tensorflow.keras.preprocessing.image import random_shear
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def shear_image(image, intensity):
    """Randomly shears an image"""
    sheared = random_shear(image, intensity, row_axis=0, col_axis=1, channel_axis=2)

    return sheared
