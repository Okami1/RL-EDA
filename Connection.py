# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:24:05 2019

@author: thoma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:07:21 2019

@author: ingmar
"""

from FitnessFunctions import OneMax, ZeroMax, LeadingOnes, BinVal

from math import log
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['image.aspect'] = 'auto'
mpl.rcParams['image.interpolation'] = 'none'

class Adam(object):
    '''
        Implements ADAM stochastic gradient descent (after Kingma & Welling, 2014)
        
        Parameter values for beta1, beta2, epsilon are as recommended in the paper
    '''
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # store parameters
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # initialize time, first and second moment 
        self.t = 0
        self.m = 0
        self.v = 0
    
    def step(self, raw_grad):
        # update time
        self.t += 1
        # effective learning rate
        alphat = self.alpha * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        # treat first steps separately
        self.m = self.beta1 * self.m + (1 - self.beta1) * raw_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * raw_grad**2
        
        return alphat * self.m / (np.sqrt(self.v) + self.epsilon)
    
class SGA(object):
    '''
        Implements stochastic gradient ascent
    '''
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def step(self, raw_grad):
        return alpha * raw_grad
    

def stoch_grad_asc(func, optim, N=100, maxiter=1000, alpha=0.01, m=10, r_past_ema=0.0,
                   r_z_score=False):
    '''
        Maximize bit-string function by stochastic gradient ascent. The returned
        accuracy assumes that the optimium is the (1, 1, ..., 1) value
        
        Args:
            func:        bit-string function to be maxmized
            N:           number of bits
            maxiter:     maximum number of iterations
            alpha:       learning rate
            m:           batchsize
            r_past_ema:  subtract exponential mean average of batch-averaged past rewards from 
                         current reward: this might help to stabilize the variance
                         of the gradient. The value should be between 0.0 and 1.0
            r_z_score:   If true, z-score rewards over batch. This both stabilizes
                         the variance and ensures that the learning rate does not 
                         get too high
    '''
    
    theta_optim = optim(alpha)
    
    # Initialize policy
    theta = np.zeros((N, maxiter))
    
    # Expected accuracy is .5
    acc = np.zeros(maxiter) + .5
    
    # Best so far fitness
    best = np.zeros(maxiter)

    for iter in range(1, maxiter):
        # calculate probabilities 1/(1+exp(-theta))
        p = 1 / (1 + np.exp(-theta[:, iter-1]))
        # calculate expected accuracy
        acc[iter-1] = np.mean(p)
        
        # generate m x N samples with probability prod()
        a = 1 * (np.random.rand(m, N) < np.tile(p[None, :], (m, 1)))
        # collect rewards
        r = func.eval2d(a)

        # update best fitness if applicable
        iteration_best = np.amax(r)
        best[iter-1] = max(iteration_best, best[iter-2])
        
        if iter == 1:
            deltar = r
        else:
            deltar = (1 - r_past_ema) * r + r_past_ema * r_past
        
        if r_z_score:
            deltar = (deltar - deltar.mean(axis=0)) / (deltar.std(axis=0) + 1e-5)
            
        # gradient step
        theta_grad = np.mean((a - p[None, :]) * deltar[:, None], axis=0)
        
        theta[:, iter] = theta[:, iter-1] + theta_optim.step(theta_grad)
        
        r_past = r.mean()
    
    # final accuracy & best fitness
    acc[iter] = np.mean(p)
    best[iter] = max(iteration_best, best[iter-1])
        
    return acc, best, theta
    
N = [100, 100, 100, 100]
#m = 1.5*N
maxiter=1000
alpha = 0.1
r_past_ema = 0.0

fitness = OneMax()

acc = []

best = np.zeros((len(N), maxiter))
theta = []

iterations = 1

#Theta drift 
_, _, theta1 = stoch_grad_asc(fitness, SGA, N=N[0], m=int(log(N[0])), maxiter=maxiter, alpha=0.1, r_past_ema=0.0, r_z_score=True) 
_, _, theta2 = stoch_grad_asc(fitness, SGA, N=N[0], m=int(0.5*N[0]), maxiter=maxiter, alpha=0.1, r_past_ema=0.0, r_z_score=True)
_, _, theta3 = stoch_grad_asc(fitness, SGA, N=N[0], m=int(0.75*N[0]), maxiter=maxiter, alpha=0.1, r_past_ema=0.0, r_z_score=True)
_, _, theta4 = stoch_grad_asc(fitness, SGA, N=N[0], m=int(N[0]), maxiter=maxiter, alpha=0.1, r_past_ema=0.0, r_z_score=True)
theta.append(theta1)
theta.append(theta2)
theta.append(theta3)
theta.append(theta4)

titles = [['N=100, m=logN', 'N=100, m=0.5N'], ['N=100, m=0.75N', 'N=100, m=N']]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

for i in range(len(axs)):
    for j in range(len(axs[i])):
        axs[i][j].imshow(theta[i])
        axs[i][j].set_title(titles[i][j])
        axs[i][j].set_xlabel('Iterations')
        axs[i][j].set_ylabel('theta_i')

plt.savefig("Convergence/Fitness_basic/Theta_drifts_Bad")
