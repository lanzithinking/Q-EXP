#!/usr/bin/env python3
"""
Spherical Slice sampling algorithm
----------------------------------
Shiwei Lan @ ASU, 2022
Created Sep 20, 2022
"""
import numpy as np

def S3(u0,l0,prior,loglik):
    '''
    Spherical Slice sampling algorithm for models with q-exponential process qEP(0, C) prior
    ----------------------------------------------------------------------------------------
    C = L * L'; S ~ unif(S^(d+1)); R^q ~ chisq(d)
    if u ~ qEP(0, C) then u = R*L*S
    ---------------------------------------------
    inputs:
      u0: initial state of the parameters
      l0: initial log-likelihood
      prior: an object containing two methods
          normalize: normalize function r(u)^(-.5)*L^(-1)u, r(u) = ||u||^2_C
          sample: random sampler generator with optional parameter S
      loglik: log-likelihood function
    outputs:
      u: new state of the parameter following qED(0,C)*lik
      l: new log-likelihood
    '''
    # choose a circle
    S = np.random.randn(u0.shape)
    S /= np.linalg.norm(S,axis=0)
    
    # log-likelihood threshold (defines a slice)
    logy = l0 + np.log(np.random.rand())
    
    # draw a initial proposal, also defining a bracket
    t = 2*np.pi*np.random.rand()
    t_min = t-2*np.pi; t_max = t;
    
    # repeat slice procedure until a proposal is accepted (land on the slice)
    S0 = prior.normalize(u0)
    R = np.random.chisquare(df=S.shape[0])
    while 1:
        S_t = S0 * np.cos(t) + S * np.sin(t)
        u = R*prior.act(S_t/np.linalg.norm(S_t,axis=0),0.5)
        # u = prior.sample(S=S_t/np.linalg.norm(S_t,axis=0))
        l = loglik(u)
        if l > logy:
            return q, l
        else:
            # shrink the bracket and try a new point
            if t < 0:
                t_min = t
            else:
                t_max = t
            t = t_min + (t_max-t_min) * np.random.rand()
