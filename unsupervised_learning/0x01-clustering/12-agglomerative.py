#!/usr/bin/env python3
"""Contains the function agglomerative()"""

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset

    Args:
        X (ndarray)(n,d): dataset
        dist (int): maximum cophenetic distance for all clusters

    Returns:
        clss (ndarray)(n,): contains cluster indices for each data point
    """
    sch = scipy.cluster.hierarchy

    link = sch.linkage(X, method='ward')
    clss = sch.fcluster(Z=link, t=dist, criterion='distance')
    dend = sch.dendrogram(link, color_threshold=dist)

    plt.show()

    return clss
