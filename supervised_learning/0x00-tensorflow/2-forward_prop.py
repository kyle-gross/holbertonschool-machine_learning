#!/usr/bin/env python3
"""
Contains the function 'forward_prop'
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network.
    
    Args:
      x: placeholder for input data
      layer_sizes: list containing the number of nodes in each layer
      activations: list containing activation functions for each layer
    
    Returns:
        prediction of network in tensor form
    """
    prediction = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(x, layer_sizes[i], activations[i])
    return prediction
