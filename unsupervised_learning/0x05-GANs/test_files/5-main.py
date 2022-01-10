#!/usr/bin/env python3


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.express as px

ganTrainer = __import__('5-train_GAN').train_gan

fakeData = ganTrainer()

values = fakeData.data.storage().tolist()
fig = px.histogram(values, title="Histogram of Forged Distribution",nbins=50)
fig.show()
