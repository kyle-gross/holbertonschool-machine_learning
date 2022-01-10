#!/usr/bin/env python3
"""Contains the class Discriminator()"""

import torch.nn as nn


class Discriminator(nn.Module):
    """Defines a Discriminator"""
    def __init__(self, input_size, hidden_size, output_size):
        """Discriminator constructor"""
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward prop
        x: input data
        """
        return self.main(x)
