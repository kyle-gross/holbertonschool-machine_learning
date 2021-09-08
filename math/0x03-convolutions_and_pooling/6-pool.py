#!/usr/bin/env python3
"""Contains the function pool()"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images

    Args:
        images (numpy.ndarray),(m,h,w,c): contains images
            m: number of images
            h: height in pixels of images
            W: width in pixels of images
            c: number of channels in image
        kernel_shape (tuple),(kh,kw): contains kernel size
            kh: height of kernel
            kw: width of kernel
        stride (tuple),(sh,sw): contains stride sizes
            sh: stride height
            sw: stride width
        mode (str): type of pooling
            max: max pooling
            avg: average pooling

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1
    output = np.zeros((m, out_h, out_w, c))

    for x in range(out_h):
        for y in range(out_w):
            if mode == 'max':
                output[:, y, x, :] = np.argmax(
                    images[:, sh*x: sh*x+kh, sw*y: sw*y+kw],
                    axis = (1, 2)
                )
            if mode == 'avg':
                output[:, y, x, :] = np.average(
                    images[:, sh*x: sh*x+kh, sw*y: sw*y+kw],
                    axis = (1, 2)
                )

    return output
