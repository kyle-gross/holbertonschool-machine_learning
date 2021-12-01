#!/usr/bin/env python3
"""Contains the function kmeans()"""

import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset

    Args:
        X (ndarray)(n,d): dataset
        k (int): no. clusters

    Returns:
        C, clss
        C (ndarray)(k,d): centroid means for each cluster
        clss (ndarray)(n,): index of the cluster in C that each data point
            belongs to
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
