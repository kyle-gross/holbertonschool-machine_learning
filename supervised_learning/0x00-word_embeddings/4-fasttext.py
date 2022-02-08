#!/usr/bin/env python3
"""Contains the function fasttext_model()"""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Creates and trains a gensim fasttext model"""
    model = FastText(
        sentences=sentences, size=size, min_count=min_count,
        negative=negative, window=window, sg=(not cbow),
        iter=iterations, seed=seed, workers=workers
    )
    model.train(
        sentences=sentences, total_examples=model.corpus_count,
        total_words=model.corpus_total_words, epochs=model.epochs
    )

    return model
