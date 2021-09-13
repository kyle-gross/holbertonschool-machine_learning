#!/usr/bin/env python3
"""Contains the function pool_backward()"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural net.

    Args:
        dA (numpy.ndarray)(m,h_new,w_new,c_new): contains partial derivatives
            with respect to the output of the pooling layer.
            m: number of examples
            h_new: height of output
            w_new: width of output
            c_new: number of channels
        A_prev (numpy.ndarray)(m,h_prev,w_prev,c): contains output of prev
            h_prev: height of prev layer
            w_prev: width of prev layer
        kernel_shape (tuple)(kh,kw): contains pooling filter size
            kh: kernel height
            kw: kernel width
        stride (tuple)(sh,sw): contains stride length
            sh: stride height
            sw: stride width
        mode (str): 'max' or 'avg' - indicates pooling type

    Returns:
        partial derivatives with respect to the prev layer(dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for n in range(m):
        for x in range(h_new):
            i = sh * x
            for y in range(w_new):
                j = sw * y
                for c in range(c_new):
                    if mode == 'avg':
                        avg_dA = dA[n, x, y, c] / kh / kw
                        dA_prev[n, i: i+kh, j: j+kw, c] += (
                            np.ones((kh, kw)) * avg_dA
                        )
                    if mode == 'max':
                        A_prev_slice = A_prev[n, i: i+kh, j: j+kw, c]
                        mask = (A_prev_slice == np.max(A_prev_slice))
                        dA_prev[n, i: i+kh, j: j+kw, c] += (
                            mask * dA[n, x, y, c]
                        )
    return dA_prev
