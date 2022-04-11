#!/usr/bin/env python3
"""Contains the function change_brightness()"""

from tensorflow.image import random_brightness


def change_brightness(image, max_delta):
    """Randomly changes the brightness of an image"""
    bright = random_brightness(image, max_delta)

    return bright
