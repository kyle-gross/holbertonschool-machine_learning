#!/usr/bin/env python3
"""Contains the function question_answer()"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to answer
    a question.
    Args:
        question [str]: contains the question to answer
        reference [str]: contains the reference document from which to
            find the answer
    Returns:
        answer [str]: contains the answer
            * If no answer, returns None
    """
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    q_tokens = tokenizer.tokenize(question)
    r_tokens = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + q_tokens + ['[SEP]'] + r_tokens + ['[SEP]']
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(q_tokens) + 1) + [1] * (len(r_tokens) + 1)
    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )

    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    if question in answer or answer == '':
        answer = None

    return answer
