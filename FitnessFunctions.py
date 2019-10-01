# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:41:51 2019

@author: thoma
"""
import numpy as np

class OneMax:
    
    def eval1d(sample):
        return np.sum(sample)
    
    def eval2d(samples):
        return np.sum(samples, axis=1)
    
    def best(n):
        return n

class LeadingOnes:
    
    def eval1d(sample):
        val = 0
        for x in sample:
            if x == 0:
                break
            val += 1
                
        return val
    
    def eval2d(samples):
        vals = np.zeros(len(samples))
        for i in range(len(samples)):
            for sample in samples[i]:
                if sample == 0:
                    break
                vals[i] += 1
                
    def best(n):
        return n

class BinVal:

    def eval1d(sample):
        val = 0
        for i, x in enumerate(sample):
            val += 2 ** (len(sample) - 1 - i) * x
        return val
    
    def eval2d(samples):
        vals = np.zeros(len(samples))
        for i in range(len(samples)):
            for idx, x in enumerate(samples[i]):
                vals[i] += 2 ** (len(samples[i]) - 1 - idx) * x
    
    def best(n):
        val = 0
        for i in range(n):
            val += 2 ** (n - 1 - i)
        return val