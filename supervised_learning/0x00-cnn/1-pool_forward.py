#!/usr/bin/env python3
"""Contains the function pool_forward()"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward prop over a pooling layer of a neural network

    Args:
        A_prev (numpy.ndarray)(m,h_prev,w_prev,c_prev): output of prev
            m: number of examples
            h_prev: height of prev layer
            w_prev: width of prev layer
            c_prev: number of channels in prev layer
        kernel_shnape (tuple)(kh,kw): contains size of filter
            kh: kernel height
            kw: kernel width
        stride (tuple)(sh,sw): strides for pooling
            sh: stride height
            sw: stride width
        mode (str): 'max' or 'avg' indicates which type of pooling to use

    Returns:
        output of pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pool_h = (h_prev - kh) // sh + 1
    pool_w = (w_prev - kw) // sw + 1
    pool = np.zeros((m, pool_h, pool_w, c_prev))

    if mode == 'max':
        func = np.amax
    if mode == 'avg':
        func = np.average

    for x in range(pool_h):
        for y in range(pool_w):
            pool[:, x, y, :] = func(
                A_prev[:, x*sh: x*sh+kh, y*sw: y*sw+kw, :],
                axis=(1, 2)
            )

    return pool
