#!/usr/bin/env python3
"""Module that prompts the user to ask a question, then gives an answer"""

exit = ['exit', 'quit', 'goodbye', 'bye']

while True:
    Q = input('Q: ')
    if Q.lower() in exit:
        print('A: Goodbye')
        exit()
    print('A: ')
