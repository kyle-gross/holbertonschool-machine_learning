#!/usr/bin/env python3
"""Contains the function sample_Z()"""

import torch


def sample_Z(mu, sigma, sampleType, dInputSize, gInputSize, mbatchSize=None):
    """Creates input for the generator and discriminator

    Args:
        mu: mean of distribution
        sigma: std. dev. of distribution
        sampleType: selects which model to sample for
            * "G" or "D"

    Return:
        * torch.Tensor type for both generator and discriminator
        * 0 if failure
    """
    if sampleType == 'D':
        return torch.normal(mu, sigma, (mbatchSize, dInputSize))
    elif sampleType == 'G':
        return torch.randn((dInputSize, gInputSize))
    else:
        return 0
