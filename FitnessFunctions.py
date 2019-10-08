# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:41:51 2019

@author: thoma
"""
import numpy as np

class OneMax:
    '''
        Implements the OneMax bit-string optimization, the 1 dimensional
        eval returns a scalar while the 2-dimensional eval returns an
        array of scalars.
    '''
    
    def eval1d(self, sample):
        return np.sum(sample)
    
    def eval2d(self, samples):
        return np.sum(samples, axis=1)
    
    def best(self, n):
        return n
    
class ZeroMax:
    '''
        Implements the ZeroMax bit-string optimization, the 1 dimensional
        eval returns a scalar while the 2-dimensional eval returns an
        array of scalars.
    '''
    
    def eval1d(self, sample):
        return len(sample) - np.sum(sample)
    
    def eval2d(self, samples):
        return len(samples[0]) - np.sum(samples, axis=1)
    
    def best(self, n):
        return n    

class LeadingOnes:
    '''
        Implements the LeadingOnes bit-string optimization, the 1 dimensional
        eval returns a scalar while the 2-dimensional eval returns an
        array of scalars.
    '''
    
    def eval1d(self, sample):
        val = 0
        for x in sample:
            if x == 0:
                break
            val += 1
                
        return val
    
    def eval2d(self, samples):
        vals = np.zeros(len(samples))
        for i in range(len(samples)):
            for sample in samples[i]:
                if sample == 0:
                    break
                vals[i] += 1
                
    def best(self, n):
        return n

class BinVal:
    '''
        Implements the BinVal bit-string optimization, the 1 dimensional
        eval returns a scalar while the 2-dimensional eval returns an
        array of scalars.
    '''

    def eval1d(self, sample):
        val = 0
        for i, x in enumerate(sample):
            val += 2 ** (len(sample) - 1 - i) * x
        return val
    
    def eval2d(self, samples):
        vals = np.zeros(len(samples))
        for i in range(len(samples)):
            l = len(samples[i])
            for idx, x in enumerate(samples[i]):
                vals[i] += 2 ** (l - 1 - idx) * x
    
    def best(self, n):
        val = 0
        for i in range(n):
            val += 2 ** (n - 1 - i)
        return val
    
class Jump:
    '''
        Implements the jump bit-string optimization, the 1 dimensional
        eval returns a scalar while the 2-dimensional eval returns an
        array of scalars.
    '''
    
    def __init__(self, m):
        self.m = m
        self.onemax = OneMax

    def eval1d(self, sample):
        o_max = self.onemax(sample)
        n = len(sample)
        if o_max < n - self.m or o_max == n:
            return self.m + o_max
        else:
            return n - o_max
        
    def eval2d(self, samples):
        vals = np.zeros(len(samples))
        for i in range(len(samples)):
            o_max = self.onemax(samples[i])
            n = len(samples[i])
            if o_max < n - self.m or o_max == n:
                vals[i] = self.m + o_max
            else:
                vals[i] = n - o_max
    
    def best(self, n):
        return n