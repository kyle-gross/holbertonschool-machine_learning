#!/usr/bin/env python3
"""Contains utility functions"""

import cv2
import numpy as np


def load_images(images_path, as_array=True):
    """Loads images

    Args:
        images_path (str): path to images
        as_arrray (bool): indicates how to load images
            * If True: load as numpy.ndarray
                * (m, h, w, c)
                * m: no. images
                * h, w, c: height, width, and number of channels
            * If False: load as list of numpy.ndarrays

    Returns:
        images, filenames
            images: list/numpy.ndarray of all images
            filenames: list of filenames associated with each image
    """
    import os

    paths = os.listdir(images_path)
    images, filenames = list(), list()

    for path in sorted(paths):
        image_path = images_path + '/' + path
        image = cv2.imread(image_path)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        filenames.append(path)

    if as_array:
        images = np.stack(images)

    return images, filenames
