#!/usr/bin/env python3
"""Contains the function question_answer()"""

qa = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """Answers questions from multiple reference texts"""
    exit = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')
        if question in exit:
            print('A: Goodbye')
            exit()
        reference = semantic_search(corpus_path, question)
        answer = qa(question, reference)
        if answer:
            print('A: ', answer)
        else:
            print('Sorry, I do not understand your question.')
