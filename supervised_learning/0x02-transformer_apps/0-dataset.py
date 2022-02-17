#!/usr/bin/env python3
"""Contains the class Dataset()"""

import tensorflow_datasets as tfds


class Dataset():
    """Loads and preps a dataset for machine translation"""
    def __init__(self):
        """Instantiates a Dataset object
        Attributes:
            data_train:
                contains ted_hrlr_translate/pt_to_en tf.data.Dataset train
            data_valid:
                contains ted_hrlr_translate/pt_to_en tf.data.Dataset validate
            tokenizer_pt:
                Portuguese tokenizer created from training set
            tokenizer_en:
                English tokenizer created from training set
        """
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset
        Args:
            data [tf.data.Dataset]:
                pt [tf.Tensor]: contains the Portuguese translation
                en [tf.Tensor]: contains corresponding English sentence
        Returns:
            tokenizer_pt, tokenizer_en
        """
        SubwordTextEncoder = tfds.deprecated.text.SubwordTextEncoder
        tokenizer_pt = SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15
        )
        tokenizer_en = SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en
