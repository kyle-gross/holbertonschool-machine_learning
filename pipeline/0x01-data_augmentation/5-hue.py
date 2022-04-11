#!/usr/bin/env python3
"""Contains the function change_hue()"""

from tensorflow.image import adjust_hue


def change_hue(image, delta):
    """Changes the hue of an image"""
    hue = adjust_hue(image, delta)

    return hue
