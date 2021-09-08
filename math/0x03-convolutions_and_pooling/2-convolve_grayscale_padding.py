#!/usr/bin/env python3
"""Contains the function convolve_grayscale_padding()"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding

    Args:
        images (numpy.ndarray),(m,h,w): images to perform convolution on
            m: number of images
            h: height (in pixels) of the imgages
            w: width (in pixels) of the images
        kernel (numpy.ndarray),(kh,kw): kernel for convolution
            kh: height of kernel
            kw: width of kernel
        padding (tuple),(ph,pw): padding values
            ph: padding height
            pw: padding width
    Returns:
        numpy.ndarray containing the convolved images
    """
    kern_h = kernel.shape[0]
    kern_w = kernel.shape[1]

    pad_h, pad_w = padding
    image_padded = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant'
    )

    m = images.shape[0]
    out_h = images.shape[1] + (2 * pad_h) - kern_h + 1
    out_w = images.shape[2] + (2 * pad_w) - kern_w + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                kernel * image_padded[:, i: i + kern_h, j: j + kern_w],
                axis=(1, 2)
            )

    return output
