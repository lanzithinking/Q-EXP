#!/usr/bin/env python
"""
Class definition of Besov prior for linear model.
-----------------------------------------------------------------------------
Created September 20, 2022 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The Q-EXP project"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu lanzithinking@outlook.com"

import os
import numpy as np
import scipy as sp
from scipy.stats import gennorm

# self defined modules
import os,sys
sys.path.append( "../" )
from util.qep.qEP import qEP

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class prior(qEP):
    """
    q-exponential process prior q-EP(mu,C) defined on 2d domain D.
    """
    def __init__(self,meshsz,L=None,mean=None,store_eig=False,**kwargs):
        if not hasattr(meshsz, "__len__"):
            self.meshsz=(meshsz,)*2
        else:
            self.meshsz=meshsz
        self._mesh()
        self.space=kwargs.pop('space','vec') # alternative 'fun'
        super().__init__(x=self.mesh, L=L, store_eig=store_eig, **kwargs)
        self.mean=mean
    
    def _mesh(self):
        """
        Build the mesh
        """
        # set the mesh
        xx,yy=np.meshgrid(np.linspace(0,1,self.meshsz[0]),np.linspace(0,1,self.meshsz[1]))
        self.mesh=np.stack([xx.flatten(),yy.flatten()]).T
        # print('\nThe mesh is defined.')
    
    def cost(self,u):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        """
        if self.mean is not None:
            u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        u_=u[:,None]
        if self.space=='vec': u_=self.vec2fun(u_)
        val=-self.logpdf(u_)[0]
        return val
    
    def grad(self,u):
        """
        Calculate the gradient of log-prior
        """
        if self.mean is not None:
            u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        # r=self.C_act(u,comp=-.5)**2
        r=abs(np.sum(u*self.C_act(u,comp=-1)))
        g=(self.N/2*(1-self.q/2)/r+self.q/4*r**(self.q/2-1))*2*self.C_act(u,comp=-1)
        return g
        
        
    def sample(self):
        """
        Sample a random function u ~ qEP(0,_C)
        """
        S=np.random.randn({'vec':self.L, 'fun':self.N}[self.space])
        S/=np.linalg.norm(S)
        R=np.random.chisquare(df=self.N)**(1./self.q)
        u=R*self.C_act(S,comp=0.5)
        if self.mean is not None:
            u+=self.mean
        return u
    
    
    def C_act(self,u,comp=1):
        """
        Calculate operation of C^comp on vector u: u --> C^comp * u
        """
        if np.ndim(u)>1: u=u.flatten()
          
        if comp==0:
            return u
        else:
            eigv, eigf=self.eigs()
            if comp<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
            Cu=(u if self.space=='vec' else eigf.T.dot(u) if self.space=='fun' else ValueError('Wrong space!'))*eigv**(comp)
            if self.space=='fun': Cu=eigf.dot(Cu)
            return Cu
    
    def vec2fun(self, u_vec):
        """
        Convert vector (u_i) to function (u)
        """
        eigv, eigf = self.eigs()
        u_f = eigf.dot(u_vec)
        return u_f
    
    def fun2vec(self, u_f):
        """
        Convert vector (u_i) to function (u)
        """
        eigv, eigf = self.eigs()
        u_vec = eigf.T.dot(u_f)
        return u_vec
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(2022)
    # define the prior
    meshsz=32
    prior = prior(meshsz=meshsz, store_eig=True, ker_opt='matern', q=1., l=.1, L=20, space='fun')
    # generate sample
    u=prior.sample()
    nlogpri=prior.cost(u)
    ngradpri=prior.grad(u)
    print('The negative logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(nlogpri,np.linalg.norm(ngradpri)))
    # test
    h=1e-7
    v=prior.sample()
    ngradv_fd=(prior.cost(u+h*v)-nlogpri)/h
    ngradv=ngradpri.dot(v.flatten())
    rdiff_gradv=np.abs(ngradv_fd-ngradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    # plot
    if u.shape[0]!=prior.N: u=prior.vec2fun(u)
    plt.imshow(u.reshape(prior.meshsz),origin='lower',extent=[0,1,0,1])
    plt.show()
    