#!/usr/bin/env python3
"""Contains the function convolve_grayscale()"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images with custom, same, or
    valid padding and custom stride lengths

    Args:
        images (numpy.ndarray),(m,h,w): images to perform convolution on
            m: number of images
            h: height (in pixels) of the imgages
            w: width (in pixels) of the images
        kernel (numpy.ndarray),(kh,kw): kernel for convolution
            kh: height of kernel
            kw: width of kernel
        padding (tuple/str),(ph,pw): padding values
            'same': performs same convolution
            'valid': performs valid convoltion
            ph: padding height
            pw: padding width
        stride (tuple),(sh,sw): stride length
            sh: stride height
            sw: stride width
    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = (((h - 1) * sh) + kh - h) // 2 + 1
        pad_w = (((w - 1) * sw) + kw - w) // 2 + 1
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding

    image_padded = np.pad(
        images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), 'constant'
    )

    out_h = (h + (2 * pad_h) - kh) // sh + 1
    out_w = (w + (2 * pad_w) - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                kernel * image_padded[:, sh*i: sh*i+kh, sw*j: sw*j+kw],
                axis=(1, 2)
            )

    return output
