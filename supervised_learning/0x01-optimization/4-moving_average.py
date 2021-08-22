#!/usr/bin/env python3
"""Contains the function moving_average
"""
import numpy as np


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set

    Exponentially weighted (moving) averages with bias correction.

    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average

    Returns:
        List containing the moving averages of data
    """
    v = 0
    avgs = []

    for t in range(1, len(data) + 1):
        v = beta * v + (1 - beta) * data[t - 1]
        bias_correction = 1 - (beta ** t)
        avgs.append(v / bias_correction)

    return avgs
