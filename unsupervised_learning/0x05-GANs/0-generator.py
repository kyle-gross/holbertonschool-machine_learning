#!/usr/bin/env python3
"""Contains the class Generator()"""

import torch.nn as nn


class Generator(nn.Module):
    """Defines a generator"""
    def __init__(self, input_size, hidden_size, output_size):
        """Class constructor"""
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """Forward-prop
        x: input data
        """
        return self.main(x)
