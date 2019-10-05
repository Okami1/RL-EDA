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

from FitnessFunctions import OneMax, LeadingOnes, BinVal

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


def stoch_grad_asc(func, N=100, maxiter=1000, alpha=0.01, m=10, r_past_ema=0.0,
                   r_z_score=False, do_adam=False):
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
            do_adam:     If true, use ADAM (Kingma & Welling, 2014) instead of vanilla
                         stochastic gradient ascent
    '''
    
    if do_adam:
        theta_adam = Adam(alpha=alpha)
    
    # Initialize policy
    theta = np.zeros((N, maxiter))
    
    # Expected accuracy is .5
    acc = np.zeros(maxiter) + .5
    
    # Fitness of best samples
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
        best[iter-1] = np.amax(r)
        
        if iter == 1:
            deltar = r
        else:
            deltar = (1 - r_past_ema) * r + r_past_ema * r_past
        
        if r_z_score:
            deltar = (deltar - deltar.mean(axis=0)) / (deltar.std(axis=0) + 1e-5)
            
        # gradient step
        theta_grad = np.mean((a - p[None, :]) * deltar[:, None], axis=0)
        
        if do_adam:
            theta[:, iter] = theta[:, iter-1] + theta_adam.step(theta_grad)
        else:
            theta[:, iter] = theta[:, iter-1] + alpha * theta_grad
        
        r_past = r.mean()
    
    # final accuracy & best fitness
    acc[iter] = np.mean(p)
    best[iter] = np.amax(r)
    
    return acc, best, theta
    
N=20000
m=10
maxiter=40000
alpha = 0.1
r_past_ema = 0.0

acc = []
best = []

acc.append(stoch_grad_asc(OneMax, N=N, m=1, maxiter=maxiter, alpha=alpha, r_past_ema=0.0, r_z_score=True, do_adam=False)[0])
acc.append(stoch_grad_asc(OneMax, N=N, m=2, maxiter=maxiter, alpha=alpha, r_past_ema=0.0, r_z_score=True, do_adam=False)[0])
acc.append(stoch_grad_asc(OneMax, N=N, m=3, maxiter=maxiter, alpha=alpha, r_past_ema=0.0, r_z_score=True, do_adam=False)[0])
acc.append(stoch_grad_asc(OneMax, N=N, m=5, maxiter=maxiter, alpha=alpha, r_past_ema=0.0, r_z_score=True, do_adam=False)[0])
acc.append(stoch_grad_asc(OneMax, N=N, m=7, maxiter=maxiter, alpha=alpha, r_past_ema=0.0, r_z_score=True, do_adam=False)[0])
acc.append(stoch_grad_asc(OneMax, N=N, m=9, maxiter=maxiter, alpha=alpha, r_past_ema=0.0, r_z_score=True, do_adam=False)[0])
acc.append(stoch_grad_asc(OneMax, N=N, m=10, maxiter=maxiter, alpha=alpha, r_past_ema=0.0, r_z_score=True, do_adam=False)[0])

legend = ['SGA, N = 50000, m = 1',
          'SGA, N = 50000, m = 2',
          'SGA, N = 50000, m = 3',
          'SGA, N = 50000, m = 5',
          'SGA, N = 50000, m = 7',
          'SGA, N = 50000, m = 9',
          'SGA, N = 50000, m = 10'
          ]

plt.figure()
plt.clf()
#plt.subplot(1,2,1)
plt.plot(np.ones(maxiter), 'k:')

handles = []
for acc_this in acc:
    handles.append(plt.plot(acc_this)[0])

plt.xlabel('Iterations')
plt.ylabel('Mean accuracy')

plt.legend(handles, legend)

#plt.subplot(1,2,2)
#plt.imshow(theta)
#
#plt.xlabel('Iterations')
#plt.ylabel('theta_i')
#plt.tight_layout()

    