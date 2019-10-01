# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:41:19 2019

@author: thoma
"""
from FitnessFunctions import OneMax, LeadingOnes, BinVal

import numpy as np
import random
import matplotlib.pyplot as plt
 
def plot_fitness(fit, best):
    index = np.array(range(len(fit)))
    plt.plot(index, fit, label="Best sample")
    plt.plot(index, [best for i in range(len(fit))], label="Best possible fitness")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Fitness value")
        

def UMDA(fit, n=10, lamb=50, mu=10, maxiter=100, verbose=False):      
    p = np.array([0.5] * n)
    m = 1/n
    fitness = []
    
    for t in range(maxiter):
        D = []
        for i in range(lamb):
            D.append([int(random.random() < p[j]) for j in range(n)])

        #Slow for long lists
        best_samples = sorted(D, key=lambda s: fit.eval1d(s))[-mu:]
        
        p = np.array([sum(x)/mu for x in zip(*best_samples)])
        p = np.vectorize(lambda x: max(m, min(1-m, x)))(p)
        
        fitness.append(fit.eval1d(best_samples[-1]))
        
        if fitness[-1] == fit.best(n):
            break
            
    if verbose:
        plot_fitness(fitness, fit.best(n))
            
    return (fitness, p)
        
def PBIL(fit, n=10, lamb=50, mu=10, maxiter=100, verbose=False, rho=0.9):
    p = np.array([0.5] * n)
    print(p)
    m = 1/n
    fitness = []
   
    for t in range(maxiter):
        D = []
        for i in range(lamb):
            D.append([int(random.random() < p[j]) for j in range(n)])
            
        #Slow for long lists
        best_samples = sorted(D, key=lambda s: fit.eval1d(s))[len(D)-mu:]
        f = np.array([sum(x)/mu for x in zip(*best_samples)])
        
        p = (1 - rho) * p + rho * f
        p = np.vectorize(lambda x: max(m, min(1-m, x)))(p)

        fitness.append(fit.eval1d(best_samples[-1]))
        
        if fitness[-1] == fit.best(n):
            break
        
    if verbose:
        plot_fitness(fitness, fit.best(n))
        
    return (fitness, p)
        
def MMAS(fit, n=10, lamb=50, mu=10, maxiter=100, verbose=False, rho=0.5):        
    p = np.array([0.5] * n)
    m = 1/n
    fitness = []
    
    for t in range(maxiter):
        D = []
        for i in range(lamb):
            D.append([int(random.random() < p[j]) for j in range(n)])
        
        best_sample = np.array(max(D, key=lambda x: fit.eval1d(x)))
        
        p = (1 - rho) * p + rho * best_sample
        p = np.vectorize(lambda x: max(m, min(1-m, x)))(p)

        fitness.append(fit.eval1d(best_sample))
        
        if fitness[-1] == fit.best(n):
            break
        
    if verbose:
        plot_fitness(fitness, fit.best(n))
        
    return (fitness, p)
    
def cGA(fit, n=10, lamb=50, mu=10, maxiter=100, verbose=False, K=10):       
    p = np.array([0.5] * n)
    m = 1/n
    fitness = []
    
    for t in range(maxiter):
        D = []
        for i in range(lamb):
            D.append([int(random.random() < p[j]) for j in range(n)])
        
        #Slow for long lists
        best_samples = np.array(sorted(D, key=lambda s: fit.eval1d(s))[-2:])
        
        p = p + 1/K*(best_samples[-1] - best_samples[-2])
        p = np.vectorize(lambda x: max(m, min(1-m, x)))(p)

        fitness.append(fit.eval1d(best_samples[-1]))
        
        if fitness[-1] == fit.best(n):
            break
        
    if verbose:
        plot_fitness(fitness, fit.best(n))
    
    return (fitness, p)
    
    
if __name__ == '__main__':
    fitness = OneMax
    n = 100
    lamb = 50
    mu = 10
    maxiter = 500
        
    #f, p = UMDA(fitness, n, lamb, mu, maxiter, verbose=False)
    #f, p = PBIL(fitness, n, lamb, mu, maxiter, verbose=False)
    #f, p = MMAS(fitness, n, lamb, mu, maxiter, verbose=True)
    #f, p = cGA(fitness, n, lamb, mu, maxiter, verbose=True)
