#!/usr/bin/env python3
"""Contains the class Dataset()"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """Loads and preps a dataset for machine translation"""
    def __init__(self, batch_size, max_len):
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
        data_train, data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train
        )
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

        def filter_max(x, y, max=max_len):
            """Filters data by max length"""
            return tf.logical_and(tf.size(x) <= max, tf.size(y) <= max)

        self.data_train = self.data_train.filter(filter_max)
        self.data_valid = self.data_valid.filter(filter_max)

        self.data_train = self.data_train.cache()
        size = len(list(self.data_train))
        self.data_train = self.data_train.shuffle(size)

        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_valid = self.data_valid.padded_batch(batch_size)

        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
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

    def encode(self, pt, en):
        """Encodes a translation into tokens.
        Args:
            pt [tf.Tensor]:
                Portuguese sentence
            en [tf.Tensor]:
                Corresponding English sentence
        Returns:
            pt_tokens, en_tokens
        """
        pt_tokens = (
            [self.tokenizer_pt.vocab_size] +
            self.tokenizer_pt.encode(pt.numpy()) +
            [self.tokenizer_pt.vocab_size + 1]
        )
        en_tokens = (
            [self.tokenizer_en.vocab_size] +
            self.tokenizer_en.encode(en.numpy()) +
            [self.tokenizer_en.vocab_size + 1]
        )

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode method"""
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded