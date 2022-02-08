#!/usr/bin/env python3
"""Contains the function cumulative_bleu()"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence"""
    unigrams = len(sentence)
    bp = 1

    min_ref = min([len(ref) for ref in references])
    if unigrams <= min_ref:
        bp = np.exp(1 - min_ref / unigrams)

    scores = [ngram_bleu(references, sentence, i) for i in range(1, n + 1)]

    return bp * np.exp(np.log(scores).sum() / n)


def ngram(sentence, n):
    """Returns ngrams for a sentence"""
    grams = []

    for i in range(len(sentence) - n + 1):
        gram = []
        for j in range(n):
            gram.append(sentence[i+j])
        grams.append(gram)

    return grams


def ngram_bleu(references, sentence, n):
    """Calculates the unigram BLEU score for a sentence
    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
    Returns:
        bleu: the unigram BLEU score
    """
    ngrams = ngram(sentence, n)
    total_ngrams = len(ngrams)
    total = 0

    while len(ngrams) > 0:
        gram = ngrams[0]
        count = ngrams.count(gram)
        [ngrams.pop(ngrams.index(gram)) for i in range(count)]

        max_ref = max([ngram(ref, n).count(gram) for ref in references])
        if count <= max_ref:
            total += count
        else:
            total += max_ref

    return total / total_ngrams
