#!/usr/bin/env python3
"""Contains the function train_GAN()"""

import torch


def train_gan(return_model=False, mbatchSize=512):
    """Trains a GAN model"""
    torch.manual_seed(111)
    
    steps = dInputSize = 20
    Generator = __import__('0-generator').Generator(1,16,1)
    Discriminator = __import__('1-discriminator').Discriminator(dInputSize,16,1)
    train_discriminator = __import__('3-train_discriminator').train_dis
    train_generator = __import__('4-train_generator').train_gen
    optimizer_g = torch.optim.SGD(
        Generator.parameters(), lr=1e-3, momentum=0.9
    )
    optimizer_d = torch.optim.SGD(
        Discriminator.parameters(), lr=1e-3, momentum=0.9
    )
    loss = torch.nn.BCELoss()

    for i in range(5000):
        if i % 100 == 0:
            print('{} iterations'.format(i))
        train_discriminator(
            Generator, Discriminator, dInputSize, 1, mbatchSize,
            steps, optimizer_d, loss
        )
        train_generator(
            Generator, Discriminator, 1, dInputSize, mbatchSize,
            steps, optimizer_g, loss
        )

    if return_model:
        return Generator
    else:
        latent_space_samples = torch.randn(100, 1)
        return Generator(latent_space_samples).detach()