#!/usr/bin/env python3
"""Contains the function gensim_to_keras()"""

import gensim.models


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras embedding layer"""
    return model.wv.get_keras_embedding(train_embeddings=False)
