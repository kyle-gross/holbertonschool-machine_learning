#!/usr/bin/env python3
"""
Contains the function 'create_placeholders'
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders x and y.
    """
    x = tf.placeholder(name='x', dtype=float, shape=[None, nx])
    y = tf.placeholder(name='y', dtype=float, shape=[None, classes])
    return x, y
