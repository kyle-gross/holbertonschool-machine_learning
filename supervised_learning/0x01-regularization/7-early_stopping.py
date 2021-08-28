#!/usr/bin/env python3
"""Contains the function early_stopping()"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early.

    Args:
        cost (float): current validation cost of network
        opt_cost (float): lowest recorded validation cost of network
        threshold (float): threshold used for early stopping
        patience (int): patience count used for early stopping
        count (int): how long the threshold has not been met

    Returns:
        Boolean, whether or not the network should be stopped early,
        followed by the updated count
    """
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if count < patience:
        return False, count
    else:
        return True, count
