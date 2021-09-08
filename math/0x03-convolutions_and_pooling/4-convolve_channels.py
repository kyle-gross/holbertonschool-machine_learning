#!/usr/bin/env python3
"""Contains the function convolve_channels()"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1,1)):
    """Performs a convolution on images with channels

    Args:
        images (numpy.ndarray),(m,h,w,c): contains images
            m: number of images
            h: height in pixels of images
            W: width in pixels of images
            c: number of channels in image
        kernel (numpy.ndarray),(kh,kw,c): contains kernel for convolution
            kh: height of kernel
            kw: width of kernel
        padding (tuple or str): if 'same', perform same
            if 'valid', perform valid
            if (tuple),(ph,pw):
                ph: padding for height
                pw: padding for width
        stride (tuple),(sh,sw):
            sh: stride height
            sw: stride width
    Returns:
        numpy.ndarray containing convolved images
    """
    m, h, w, c = images.shape
    kh, kw, output_d = kernel.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = (((h - 1) * sh) + kh - h) // 2 + 1
        pad_w = (((w - 1) * sw) + kw - w) // 2 + 1
    elif padding == 'valid':
        pad_h, pad_w = 0, 0
    else:
        pad_h, pad_w = padding
    image_padded = np.pad(
        images, ((0,), (pad_h,), (pad_w,), (0,)), 'constant'
    )

    out_h = (h + (2 * pad_h) - kh) // sh + 1
    out_w = (w + (2 * pad_w) - kw) // sw + 1
    output = np.zeros((m, out_h, out_w))

    for x in range(out_h):
        for y in range(out_w):
            output[:, x, y] = np.sum(
                kernel * image_padded[:, sh*x: sh*x+kh, sw*y: sw*y+kw],
                axis=(1, 2, 3)
            )

    return output
