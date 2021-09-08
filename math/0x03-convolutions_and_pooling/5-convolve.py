#!/usr/bin/env python3
"""Contains the function convolve()"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels

    Args:
        images (numpy.ndarray),(m,h,w,c): contains images
            m: number of images
            h: height in pixels of images
            W: width in pixels of images
            c: number of channels in image
        kernels (numpy.ndarray),(kh,kw,c,nc): contains kernels for convolutions
            kh: height of kernel
            kw: width of kernel
            nc: numer of kernels
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
    kh, kw, output_d, nc = kernels.shape
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
    output = np.zeros((m, out_h, out_w, nc))

    for ch in range(nc):
        for x in range(out_h):
            for y in range(out_w):
                output[:, x, y, ch] = np.sum(
                    np.multiply(
                        kernels[:, :, :, ch],
                        image_padded[:, sh*x: sh*x+kh, sw*y: sw*y+kw]
                    ), axis=(1, 2, 3)
                )

    return output
