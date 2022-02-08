#!/usr/bin/env python3
"""Contains the function tf_idf()"""

from sklearn.feature_extraction.text import TfidfVectorizer
from torch import embedding


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding
    Args:
        sentences (list): list of sentences to analyze
        vocab (list): list of vocabulary words to use for analysis
            * If None, all words within sentences should be used
    Returns:
        embeddings, features
        embeddings (ndarray)(s,f): contains embeddings
            s: no. sentences in sentences
            f: no. features analyzed
        features (list): list of features used for embeddings
    """
    vectorizer = TfidfVectorizer(lowercase=True, vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
