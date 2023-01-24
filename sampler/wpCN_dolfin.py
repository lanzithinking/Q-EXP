#!/usr/bin/env python
"""
Geometric Infinite dimensional MCMC samplers
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Created March 12, 2016
-------------------------------
Modified Dec. 8, 2021 @ ASU
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2016, The EQUIP/EQUiPS projects"
__license__ = "GPL"
__version__ = "1.3"
__maintainer__ = "Shiwei Lan"
__email__ = "S.Lan@warwick.ac.uk; lanzithinking@outlook.com; slan@asu.edu"

import numpy as np
import dolfin as df
import timeit,time

def wpCN(u0,l0,loglik,T,h=0.1):
    """
    Whitened Preconditioned Crank-Nicolson (wpCN)
    ---------------------------------------------------------
    inputs:
      u0: initial state of the parameters
      l0: initial log-likelihood\
      loglik: log-likelihood function
      T: map transforming white noise to (non)-Gaussian prior mu0
      h: step size
    outputs:
      u: new state of the parameter following lik*mu0
      l: new log-likelihood
      acpt: indicator of acceptance
    '''
    """
    # sample velocity
    v=df.Vector(u0)
    v.set_local(np.random.randn(u0.size()))
    
    # generate proposal according to Crank-Nicolson scheme
    # u = ((1-h/4)*u0 + np.sqrt(h)*v)/(1+h/4)
    u = np.sqrt(1-h**2)*u0 + h*v
    
    # update loglik
    l=loglik(T(u))
    
    # Metropolis test
    logr=l-l0
    
    if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
        # accept
        acpt=True
    else:
        # reject
        u=u0; l=l0
        acpt=False
        
    # return 
    return u,l,acpt
