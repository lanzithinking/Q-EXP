#!/usr/bin/env python3
"""
Elliptical Slice Sampler algorithm
----------------------------------
Shiwei Lan @ ASU, 2022
Created Sep 20, 2022
"""
import numpy as np

def ESS(u0,l0,rnd_pri,loglik):
    '''
    Elliptical Slice Sampler algorithm by Murray et~al (2010)
    ---------------------------------------------------------
    inputs:
      u0: initial state of the parameters
      l0: initial log-likelihood
      rnd_pri: random sample generator from the prior N(0,C)
      loglik: log-likelihood function
    outputs:
      u: new state of the parameter following N(0,C)*lik
      l: new log-likelihood
    '''
    # choose an ellipsis
    v = rnd_pri() # (u, v) now defines a slice 
    
    # log-likelihood threshold (defines a slice)
    logy = l0 + np.log(np.random.rand())
    
    # draw a initial proposal, also defining a bracket
    t = 2*np.pi*np.random.rand()
    t_min = t-2*np.pi; t_max = t;
    
    # repeat slice procedure until a proposal is accepted (land on the slice)
    while 1:
        u = u0 * np.cos(t) + v * np.sin(t)
        l = loglik(u)
        if l > logy:
            return u, l
        else:
            # shrink the bracket and try a new point
            if t < 0:
                t_min = t
            else:
                t_max = t
            t = t_min + (t_max-t_min) * np.random.rand()
