#!/usr/bin/env python3
"""Contains the function answer_loop()"""

qa = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answers questions"""
    exit = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        question = input('Q: ')
        if question in exit:
            print('A: Goodbye')
            exit()
        answer = qa(question, reference)
        if answer:
            print('A: ', answer)
        else:
            print('Sorry, I do not understand your question.')
