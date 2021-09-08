#!/usr/bin/env python3
"""Contains the function convolve_grayscale_valid()"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images

    Args:
        images (numpy.ndarray),(m,h,w): images to perform convolution on
            m: number of images
            h: height (in pixels) of the imgages
            w: width (in pixels) of the images
        kernel (numpy.ndarray),(kh,kw): kernel for convolution
            kh: height of kernel
            kw: width of kernel

    Returns:
        numpy.ndarray containing the convolved images
    """
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    out_h = images.shape[1] - kh + 1
    out_w = images.shape[2] - kw + 1
    m = images.shape[0]

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                kernel * images[:, i: i + kh, j: j + kw],
                axis=(1, 2)
            )

    return output
