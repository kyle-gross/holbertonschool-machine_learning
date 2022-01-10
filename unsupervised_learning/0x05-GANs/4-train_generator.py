#!/usr/bin/env python3
"""Contains the function train_gen()"""

import torch

sample_Z = __import__('2-sample_Z').sample_Z


def train_gen(Gen, Dis, gInputSize, dInputSize, mbatchSize, steps, optimizer, crit):
    """Trains the generator"""
    for _ in range(steps):
        # Generate data
        real_samples_labels = torch.ones((mbatchSize, 1))

        generated_list = []

        for _ in range(mbatchSize):
            latent_space_samples = sample_Z(0.0, 1.0, 'G', dInputSize, gInputSize)
            generated_samples = Gen(latent_space_samples)
            generated_list.append(generated_samples)
        
        generated_samples = torch.stack(generated_list, axis=0).reshape(mbatchSize, dInputSize)

        # Train generator
        output_discriminator_generated = Dis(generated_samples)
        loss_generator = crit(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer.step()
    
    return loss_generator, generated_samples
