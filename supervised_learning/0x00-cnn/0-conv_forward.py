#!/usr/bin/env python3
"""Contains the function conv_forward()"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer of a neural
    network.

    Args:
        A_prev (numpy.ndarray)(m,h_prev,w_prev,c_prev): output of prev
            m: number of examples
            h_prev: height of prev layer
            w_prev: width of prev layer
            c_prev: number of channels in prev layer
        W (numpy.ndarray)(kh,kw,c_prev,c_new): contains filters for conv
            kh: filter height
            kw: filter width
            c_prev: number of channels in prev layer
            c_new: number of channels in output
        b (numpy.ndarray)(1,1,1,c_new): biases applied to conv
        activation (func): activation function
        padding (str): 'same' or 'valid'
        stride (tuple)(sh,sw): holds stride values
            sh: stride height
            sw: stride width

    Returns:
        output of conv layer
    """
    sh, sw = stride
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, nc = W.shape

    if padding == 'same':
        pad_h = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pad_w = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1
    if padding == 'valid':
        pad_h, pad_w = 0, 0

    A_padded = np.pad(
        A_prev, ((0,), (pad_h,), (pad_w,), (0,)), 'constant'
    )
    conv_h = (h_prev + (2 * pad_h) - kh) // sh + 1
    conv_w = (w_prev + (2 * pad_w) - kw) // sw + 1
    conv = np.zeros((m, conv_h, conv_w, nc))

    for z in range(nc):
        for x in range(conv_h):
            for y in range(conv_w):
                conv[:, x, y, z] = np.sum(
                    np.multiply(
                        W[:, :, :, z],
                        A_padded[:, sh*x: sh*x+kh, sw*y:sw*y+kw]
                    ), axis=(1, 2, 3)
                ) + b[:, :, :, z]

    return activation(conv)
