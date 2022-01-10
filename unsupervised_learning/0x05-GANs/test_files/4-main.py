#!/usr/bin/env python3


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

genTrain = __import__('4-train_generator').train_gen

learning_rate = 1e-3
sgd_momentum = 0.9
g_input_size = 1   
g_hidden_size = 5  
g_output_size = 1  
minibatch_size =  500 
d_input_size = 1 #here 4 or 1
d_hidden_size = 10 
d_output_size = 1  
num_epochs = 50
d_steps = 20
g_steps = 20



G_Test = __import__('0-generator').Generator(1,5,1)
D_Test = __import__('1-discriminator').Discriminator(1,10,1)



dOpt_Test = optim.SGD(D_Test.parameters(), lr=learning_rate, momentum=sgd_momentum)
crit_Test = nn.BCELoss()  


print(genTrain(G_Test,D_Test,g_input_size, minibatch_size, g_steps, dOpt_Test, crit_Test))
