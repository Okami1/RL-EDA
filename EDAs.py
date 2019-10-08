# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:41:19 2019

@author: thoma
"""
from FitnessFunctions import OneMax, LeadingOnes, BinVal

import numpy as np
import random
import matplotlib.pyplot as plt    

def UMDA(fit, n=10, lamb=50, mu=10, maxiter=100):
    '''
        Maximize bit-string function by the UMDA EDA.
        Returns the best fitness over time along with the final probability vector.
    '''
    p = np.zeros((maxiter, n))
    p[0] = np.array([0.5] * n)
    m = 1/n
    fitness = []
    
    for i in range(1, maxiter):
        D = []
        for _ in range(lamb):
            D.append([int(random.random() < p[i-1][j]) for j in range(n)])

        #Slow for long lists
        best_samples = sorted(D, key=lambda s: fit.eval1d(s))[-mu:]
        
        p[i] = np.array([sum(x)/mu for x in zip(*best_samples)])
        p[i] = np.vectorize(lambda x: max(m, min(1-m, x)))(p[i])
        
        fitness.append(fit.eval1d(best_samples[-1]))
        
        if fitness[-1] == fit.best(n):
            break

    return (fitness, p)
        
def PBIL(fit, n=10, lamb=50, mu=10, maxiter=100, rho=0.9):
    '''
        Maximize bit-string function by the PBIL EDA.
        Returns the best fitness over time along with the final probability vector.
    '''
    p = np.zeros((maxiter, n))
    p[0] = np.array([0.5] * n)
    m = 1/n
    fitness = []
   
    for i in range(1, maxiter):
        D = []
        for _ in range(lamb):
            D.append([int(random.random() < p[i-1][j]) for j in range(n)])
            
        #Slow for long lists
        best_samples = sorted(D, key=lambda s: fit.eval1d(s))[len(D)-mu:]
        f = np.array([sum(x)/mu for x in zip(*best_samples)])
        
        p[i] = (1 - rho) * p[i-1] + rho * f
        p[i] = np.vectorize(lambda x: max(m, min(1-m, x)))(p[i])

        fitness.append(fit.eval1d(best_samples[-1]))
        
        if fitness[-1] == fit.best(n):
            break
        
    return (fitness, p)
        
def MMAS(fit, n=10, lamb=50, mu=10, maxiter=100, rho=0.5):
    '''
        Maximize bit-string function by the MMAS EDA.
        Returns the best fitness over time along with the final probability vector.
    '''
    p = np.zeros((maxiter, n))
    p[0] = np.array([0.5] * n)
    m = 1/n
    fitness = []
    
    for i in range(1, maxiter):
        D = []
        for _ in range(lamb):
            D.append([int(random.random() < p[i-1][j]) for j in range(n)])
        
        best_sample = np.array(max(D, key=lambda x: fit.eval1d(x)))
        
        p[i] = (1 - rho) * p[i-1] + rho * best_sample
        p[i] = np.vectorize(lambda x: max(m, min(1-m, x)))(p[i])

        fitness.append(fit.eval1d(best_sample))
        
        if fitness[-1] == fit.best(n):
            break
        
    return (fitness, p)
    
def cGA(fit, n=10, lamb=50, mu=10, maxiter=100, K=10):
    '''
        Maximize bit-string function by the CGA EDA.
        Returns the best fitness over time along with the final probability vector.
    '''
    p = np.zeros((maxiter, n))
    p[0] = np.array([0.5] * n)
    m = 1/n
    fitness = []
    
    for i in range(1, maxiter):
        D = []
        for _ in range(lamb):
            D.append([int(random.random() < p[i-1][j]) for j in range(n)])
        
        #Slow for long lists
        best_samples = np.array(sorted(D, key=lambda s: fit.eval1d(s))[-2:])
        
        p[i] = p[i-1] + 1/K*(best_samples[-1] - best_samples[-2])
        p[i] = np.vectorize(lambda x: max(m, min(1-m, x)))(p[i])

        fitness.append(fit.eval1d(best_samples[-1]))
        
        if fitness[-1] == fit.best(n):
            break
    
    return (fitness, p)
    
    
if __name__ == '__main__':
    fitness = OneMax()
    n = 100
    lamb = 50
    mu = 10
    maxiter = 500
        
    #f, p = UMDA(fitness, n, lamb, mu, maxiter)
    #f, p = PBIL(fitness, n, lamb, mu, maxiter)
    f, p = MMAS(fitness, n, lamb, mu, maxiter)
    #f, p = cGA(fitness, n, lamb, mu, maxiter)
    print(f)
    print(p)