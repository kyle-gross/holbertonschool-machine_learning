#!/usr/bin/env python3
"""Contains the function convolve_grayscale_same()"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images

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
    kh, kw = kernel.shape
    m, out_h, out_w = images.shape

    pad_h = kh // 2
    pad_w = kw // 2
    image_padded = np.pad(
        images, ((0,), (pad_h,), (pad_w,)), 'constant'
    )

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                kernel * image_padded[:, i: i + kh, j: j + kw],
                axis=(1, 2)
            )

    return output
