#!/usr/bin/env python3
"""Contains the function semantic_search()"""

import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents.
    Args:
        corpus_path [str]: path to corpus os reference documents to search
        sentence [str]: sentence to perform semantic search on
    Returns:
        reference [str]: the reference text of the document most similar to
            `sentence`.
    """
    model = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5'
    )
    document = [sentence]

    for fn in os.listdir(corpus_path):
        if fn.endswith('.md'):
            with open(corpus_path + '/' + fn, 'r') as f:
                document.append(f.read())

    embeddings = model(document)
    correlation = np.inner(embeddings, embeddings)
    closest = np.argmax(correlation[0, 1:])

    reference = document[closest + 1]

    return reference
