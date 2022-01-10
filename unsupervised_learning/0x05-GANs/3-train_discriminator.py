#!/usr/bin/evn python3
"""Contains the function train_dis()"""

import torch

sample_Z = __import__('2-sample_Z').sample_Z


def train_dis(Gen, Dis, dInputSize, gInputSize, mbatchSize, steps, optimizer,
              crit):
    """Trains a discriminator

    Args:
        Gen: Generator
        Dis: Discriminator
        dInputSize: input size of Dis. data
        gInputSize: input size of Gen. data
        mbatchSize: batch size for training
        steps: # of steps for training
        optimizer: stochastic gradient descent optimizer object
        crit: BCEloss function

    Return:
        Error estimate of the fake and real data, along with fake and real
            data sets of type torch.Tensor()
    """

    for _ in range(steps):
        # Discriminator data
        real_samples = sample_Z(
            0.0, 1.0, 'D', dInputSize, gInputSize, mbatchSize
        )
        real_samples_labels = torch.ones((mbatchSize, 1))

        generated_list = []

        for _ in range(mbatchSize):
            latent_space_samples = sample_Z(
                0.0, 1.0, 'G', dInputSize, gInputSize
            )
            generated_samples = Gen(latent_space_samples)
            generated_list.append(generated_samples)
        
        generated_samples = torch.stack(generated_list, axis=0).reshape(mbatchSize, dInputSize)
        generated_samples_labels = torch.zeros((mbatchSize, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        Dis.zero_grad()
        output_discriminator = Dis(all_samples)
        loss_discriminator = crit(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer.step()

    return loss_discriminator, all_samples
