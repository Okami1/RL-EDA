"""
Created on Wed Jul 17 15:07:21 2019

@author: ingmar
"""

from FitnessFunctions import OneMax, ZeroMax, LeadingOnes, BinVal, Jump

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
        return self.alpha * raw_grad
    

def stoch_grad_asc(func, optim, N=100, maxiter=1000, alpha=0.01, m=10, r_past_ema=0.0, r_z_score=False, margins=False):
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
        Returns:
            acc:         array of expected accuracy over time
            best:        array of best fitness over time
            theta:       2d-array of theta over time
            drift:       array of number of probabilities < 0.3 over time
    '''
    
    if margins:
        margin = 1/N
    theta_optim = optim(alpha)
    
    # Initialize policy
    theta = np.zeros((N, maxiter))
    
    # Expected accuracy is .5
    acc = np.zeros(maxiter) + .5
    
    # Best so far fitness
    best = np.zeros(maxiter)
    
    # Count drifts
    drift = np.zeros(maxiter)

    for iter in range(1, maxiter):
        # calculate probabilities 1/(1+exp(-theta))
        p = 1 / (1 + np.exp(-theta[:, iter-1]))
        if margins:
            p = np.vectorize(lambda x: max(margin, min(1-margin, x)))(p) #Margin
        
        # calculate expected accuracy
        acc[iter-1] = np.mean(p)
        
        # generate m x N samples with probability prod()
        a = 1 * (np.random.rand(m, N) < np.tile(p[None, :], (m, 1)))
        
        # collect rewards
        r = func.eval2d(a)

        # update best fitness if applicable
        iteration_best = np.amax(r)
        best[iter-1] = max(iteration_best, best[iter-2])
        
        # update probabilities
        drift[iter-1] = (p <= 0.3).sum()
        
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
    drift[iter] = (p <= 0.3).sum()
    
    
    return acc, best, theta, drift

### Set parameters ###
tests = 7
num_exp = 10
maxiter = 5000

N = 100
#N = np.array([400, 600, 800, 1000, 1200, 1400, 1600])


fitness = OneMax()
'''
k = 0.1*N
fitness = Jump(k)
'''

m = N
#math.log does not work on arrays, uncomment this and index m values in function calls instead
#m = [N*log(N), N*log(N), N*log(N), N*log(N), N*log(N), N*log(N), N*log(N)]


alpha = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015]

r_past_ema = 0.0
r_z_score = False
margins = False

'''
### Compute theta drift ###
drift = np.zeros((tests, maxiter))
for i in range(numExperiments):
    print("# Experiment: ", i+1)
    drift[0] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[0], r_past_ema=0.0, r_z_score=r_z_score, margins=margins)[3]
    drift[1] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[1], r_past_ema=0.0, r_z_score=r_z_score, margins=margins)[3]
    drift[2] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[2], r_past_ema=0.0, r_z_score=r_z_score, margins=margins)[3]
    drift[3] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[3], r_past_ema=0.0, r_z_score=r_z_score, margins=margins)[3]
    drift[4] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[4], r_past_ema=0.0, r_z_score=r_z_score, margins=margins)[3]
    drift[5] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[5], r_past_ema=0.0, r_z_score=r_z_score, margins=margins)[3]
    drift[6] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[6], r_past_ema=0.0, r_z_score=r_z_score, margins=margins)[3]
drift /= numExperiments

'''
### Compute Convergence ###
best = np.zeros((tests, maxiter))
for i in range(num_exp):
    print("# Experiment: ", i+1)
    best[0] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[0], r_past_ema=r_past_ema, r_z_score=r_z_score, margins=margins)[1]
    best[1] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[1], r_past_ema=r_past_ema, r_z_score=r_z_score, margins=margins)[1]
    best[2] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[2], r_past_ema=r_past_ema, r_z_score=r_z_score, margins=margins)[1]
    best[3] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[3], r_past_ema=r_past_ema, r_z_score=r_z_score, margins=margins)[1]
    best[4] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[4], r_past_ema=r_past_ema, r_z_score=r_z_score, margins=margins)[1]
    best[5] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[5], r_past_ema=r_past_ema, r_z_score=r_z_score, margins=margins)[1]
    best[6] += stoch_grad_asc(fitness, SGA, N=N, m=int(m), maxiter=maxiter, alpha=alpha[6], r_past_ema=r_past_ema, r_z_score=r_z_score, margins=margins)[1]
best /= num_exp


### Print Results ###
legend = [f'N = {N}, m = N, a = {alpha[0]}, m/a = {m/alpha[0]}',
          f'N = {N}, m = N, a = {alpha[1]}, m/a = {m/alpha[1]}',
          f'N = {N}, m = N, a = {alpha[2]}, m/a = {m/alpha[2]}',
          f'N = {N}, m = N, a = {alpha[3]}, m/a = {m/alpha[3]}',
          f'N = {N}, m = N, a = {alpha[4]}, m/a = {m/alpha[4]}',
          f'N = {N}, m = N, a = {alpha[5]}, m/a = {m/alpha[5]}',
          f'N = {N}, m = N, a = {alpha[6]}, m/a = {m/alpha[6]}',
          ]


'''
for i in range(len(N)):             #Plot lines for multiple values of N
    plt.plot(np.ones(maxiter) * N[i])
'''
plt.plot(np.ones(maxiter) * (N))    #Plot line for single N value

handles = []
for best_this in best:
    handles.append(plt.plot(best_this)[0])

plt.figure(figsize=(8,8))
plt.xlabel('Iterations')
plt.ylabel('Best fitness')
plt.legend(handles, legend)
#plt.savefig(f"Graphs/Jump/Convergence/1_10_100_quadratic.png")
