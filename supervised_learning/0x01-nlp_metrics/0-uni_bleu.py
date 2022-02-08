#!/usr/bin/env python3
"""Contains the function uni_bleu()"""

import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence
    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
    Returns:
        bleu: the unigram BLEU score
    """
    unigrams = len(sentence)
    bp = 1
    total = 0

    min_ref = min([len(ref) for ref in references])
    if unigrams <= min_ref:
        bp = np.exp(1 - min_ref / unigrams)

    while len(sentence) > 0:
        # Only count each unique word once
        word = sentence[0]
        count = sentence.count(word)
        [sentence.pop(sentence.index(word)) for i in range(count)]

        max_ref = max([ref.count(word) for ref in references])
        if count <= max_ref:
            total += count
        else:
            total += max_ref

    return bp * (total / unigrams)
