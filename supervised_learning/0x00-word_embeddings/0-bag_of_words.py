#!/usr/bin/env python3
"""Contains the function bag_of_words()"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix
    Args:
        sentences (list): list of sentences to analyze
        vocab (list): list of vocabulary words to use for analylsis
    Returns:
        embeddings, features
        embeddings (ndarray)(s,f): contains embeddings
            s: no. sentences in `sentences`
            f: no. features analyzed
        features (list): list of features used for embeddings
    """
    vectorizer = CountVectorizer(lowercase=True, vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
