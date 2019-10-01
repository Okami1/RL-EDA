# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:18:54 2019

@author: thoma
"""
from FitnessFunctions import OneMax, LeadingOnes, BinVal

import random
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

#PARAMETERS
N = 2
maxiter = 1
m = 5
T = 5
alpha = 0.1
fitness = OneMax
discount_factor = 0.5


# define network
class Net(nn.Module):

    def __init__(self, num_features, num_hidden, num_output):
        super(Net, self).__init__()  
        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(num_hidden, num_features)))
        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden)))
        
        self.activation = torch.relu
        
    def forward(self, x):
        x = F.linear(x, self.W_1)
        x = self.activation(x)
        x = F.linear(x, self.W_2)
        return F.softmax(x, dim=0)

hidden = 5
num_weights = 2*N*hidden
net = Net(N, hidden, N)

for i in range(maxiter):
    theta_r = torch.zeros(num_weights)
    
    for j in range(m):
        gradients = torch.zeros(num_weights, T)
        rewards = torch.zeros(T)
        
        # Define start state
        s = torch.Tensor([int(random.random() < 0.5) for k in range(N)])
        
        for j in range(T):
            # Compute probabilities
            p = net(s)
                
            # Greedy bit flip
            prob, idx = torch.max(p, 0)
            s.data[idx] = int(not s.data[idx])
            
            torch.log(prob).backward()
                
            g_W1 = torch.reshape(net.W_1.grad, [-1])
            g_W2 = torch.reshape(net.W_2.grad, [-1])
            g_theta = theta = torch.cat((g_W1, g_W2))
            gradients[:, j] = g_theta
            
            rewards[j] = fitness(s)
        
        expected_rewards = torch.Tensor([ sum([ (discount_factor ** (k-j)) * rewards[k] for k in range(j, T) ]) for j in range(T)])
        
        theta_r += torch.matmul(gradients, expected_rewards)
    
    theta_r = alpha * 1/m * theta_r
    
    W1 = torch.reshape(net.W_1, [-1])
    W2 = torch.reshape(net.W_2, [-1])
    theta = torch.cat((W1, W2))
    
    theta = theta + theta_r
    #Transfer theta_r back to the weights        
        
