#!/usr/bin/env python3
"""Contains the function conv_backward()"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs back prop over a convolutional layer of a neural net

    Args:
        dZ (numpy.ndarray)(m,h_new,w_new,c_new): contains partial derivatives
            with respect to unactivated output of the conv layer
            m: number of examples
            h_new: height of output
            w_new: width of output
            c_new: number of channels in output
        A_prev (numpy.ndarray)(m,h_prev,w_prev,c_prev): output of prev layer
            h_prev: height of prev layer
            w_prev: width of prev layer
            c_prev: number of channels in prev layer
        W (numpy.ndarray)(kh,kw,c_prev,c_new): contains filters
            kh: filter height
            kw: filter width
        b (numpy.ndarray)(1,1,1,c_new): contains biases applied to conv
        padding (str): 'same' or 'valid' padding style
        stride (tuple)(sh,sw): contains stride info
            sh: stride height
            sw: stride width

        Returns:
            partial derivatives with respect to prev layer(dA_prev),
            filters(dW), and biases(db), respectively.
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == 'same':
        pad_h = (((h_prev - 1) * sh) + kh - h_prev) // 2 + 1
        pad_w = (((w_prev - 1) * sw) + kw - w_prev) // 2 + 1
    if padding == 'valid':
        pad_h, pad_w = 0, 0
    A_padded = np.pad(
        A_prev, ((0,), (pad_h,), (pad_w,), (0,)), 'constant'
    )
    dA_padded = np.pad(
        dA_prev, ((0,), (pad_h,), (pad_w,), (0,)), 'constant'
    )

    for n in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for z in range(c_new):
                    dW[:, :, :, z] += (
                        np.multiply(
                            A_padded[n, sh*x: sh*x+kh, sw*y: sw*y+kw, :],
                            dZ[n, x, y, z]
                        )
                    )
                    dA_padded[n, sh*x: sh*x+kh, sw*y: sw*y+kw, :] += (
                        np.multiply(
                            W[:, :, :, z],
                            dZ[n, x, y, z]
                        )
                    )

    if padding == 'same':
        dA_prev = dA_padded[:, pad_h: -pad_h, pad_w: -pad_w, :]
    else:
        dA_prev = dA_padded

    return dA_prev, dW, db
