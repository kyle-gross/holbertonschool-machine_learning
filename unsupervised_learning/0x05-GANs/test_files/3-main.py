#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

disTrain = __import__('3-train_discriminator').train_dis

learning_rate = 1e-3
sgd_momentum = 0.9
g_input_size = 1   
g_hidden_size = 5  
g_output_size = 1 # here 1 or 4
minibatch_size =  500 
d_input_size = 1 # here 1 or 4
d_hidden_size = 10 
d_output_size = 1  
num_epochs = 50
d_steps = 20
g_steps = 20

G_Test = __import__('0-generator').Generator(1,5,1) # 3rd arg 1 or 4
D_Test = __import__('1-discriminator').Discriminator(1,10,1) # 3rd arg 1 or 4



opt_Test = optim.SGD(G_Test.parameters(), lr=learning_rate, momentum=sgd_momentum)
crit_Test = nn.BCELoss() 

print(disTrain(G_Test,D_Test, d_input_size,g_input_size, minibatch_size, d_steps, opt_Test, crit_Test))
