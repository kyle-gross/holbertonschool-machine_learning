#!/usr/bin/env python3
"""Contains the function word2vec_model()"""

from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Creates and trains a genism word2vec model
    Args:
        sentences (list): sentences to train on
        size (int): dimensionality of the embedding layer
        min_count (int): min. no. occurrences of a word to use in training
        window (int): max distance between current and predicted word
        negative (int): size of negative sampling
        cbow (bool): determines training type
            * True for CBOW
            * False for Skip-gram
        iterations (int): no. iterations to train over
        seed (int): seed for random number generator
        workers (int): no. worker threads to train model
    Returns:
        trained model
    """
    model = Word2Vec(
        sentences=sentences, size=size, min_count=min_count, window=window,
        negative=negative, sg=(not cbow), iter=iterations, seed=seed,
        workers=workers
    )
    model.train(
        sentences, total_examples=model.corpus_count, epochs=model.epochs,
        total_words=model.corpus_total_words
    )

    return model
