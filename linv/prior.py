#!/usr/bin/env python
"""
Class definition of prior for linear model.
--------------------------------------------------------------------------
Created February 15, 2022 for project of Spatiotemporal Besov prior (STBP)
updated September 20, 2022 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project and the Q-EXP project"
__license__ = "GPL"
__version__ = "0.4"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu lanzithinking@outlook.com"

import os
import numpy as np
import scipy as sp
from scipy.stats import gennorm

# self defined modules
import os,sys
sys.path.append( "../" )
from util.gp.GP import GP
from util.bsv.BSV import BSV
from util.qep.qEP import qEP

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class _gp(GP):
    """
    Gaussian process prior GP(mu,C) defined on 2d domain D.
    """
    def __init__(self,input,L=None,mean=None,store_eig=False,**kwargs):
        x=self._mesh(imsz=input) if np.size(input)<=2 and not isinstance(input,str) else input
        self.space=kwargs.pop('space','vec') # alternative 'fun'
        super().__init__(x=x, L=L, store_eig=store_eig, **kwargs)
        self.mean=mean
    
    def _mesh(self,imsz=None):
        """
        Build the mesh
        """
        if not hasattr(imsz, "__len__"):
            self.imsz=(imsz,)*2
        else:
            self.imsz=imsz
        # set the mesh
        xx,yy=np.meshgrid(np.linspace(0,1,self.imsz[0]),np.linspace(0,1,self.imsz[1]))
        mesh=np.stack([xx.flatten(),yy.flatten()]).T
        # print('\nThe mesh is defined.')
        return mesh
    
    def cost(self,u):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        """
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        u_=u[:,None]
        if self.space=='vec': u_=self.vec2fun(u_)
        val=-self.logpdf(u_)[0]
        return val
    
    def grad(self,u):
        """
        Calculate the gradient of log-prior
        """
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        g=self.C_act(u,comp=-1)
        return g
        
    def sample(self):
        """
        Sample a random function u ~ N(0,_C)
        """
        Z=np.random.randn({'vec':self.L, 'fun':self.N}[self.space])
        u=self.C_act(Z,comp=0.5)
        if self.mean is not None:
            u+=self.mean
        return u
    
    def C_act(self,u,comp=1):
        """
        Calculate operation of C^comp on vector u: u --> C^comp * u
        """
        if self.mean is not None: u-=self.mean
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

class _bsv(BSV):
    """
    Besov prior measure B(mu,_C) defined on 2d domain D.
    """
    def __init__(self,input,L=None,mean=None,store_eig=False,**kwargs):
        x=self._mesh(imsz=input) if np.size(input)<=2 and not isinstance(input,str) else input
        self.space=kwargs.pop('space','vec') # alternative 'fun'
        super().__init__(x=x, L=L, store_eig=store_eig, **kwargs)
        self.mean=mean
    
    def _mesh(self,imsz=None):
        """
        Build the mesh
        """
        if not hasattr(imsz, "__len__"):
            self.imsz=(imsz,)*2
        else:
            self.imsz=imsz
        # set the mesh
        xx,yy=np.meshgrid(np.linspace(0,1,self.imsz[0]),np.linspace(0,1,self.imsz[1]))
        mesh=np.stack([xx.flatten(),yy.flatten()]).T
        # print('\nThe mesh is defined.')
        return mesh
    
    def cost(self,u):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        -.5* ||_C^(-1/q) u(x)||^q = -.5 sum_l |gamma_l^{-1} <phi_l, u>|^q
        """
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        proj_u=self.C_act(u, -1.0/self.q, proj=True)
        
        val=0.5*np.sum(abs(proj_u)**self.q)
        return val
    
    def grad(self,u):
        """
        Calculate the gradient of log-prior
        """
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        eigv, eigf=self.eigs()
        if self.space=='vec':
            proj_u=u/eigv**(1/self.q)
            g=0.5*self.q*abs(proj_u)**(self.q-1) *np.sign(proj_u)/eigv**(1/self.q)
        elif self.space=='fun':
            proj_u=eigf.T.dot(u)/eigv**(1/self.q)
            g=eigf.dot(0.5*self.q*abs(proj_u)**(self.q-1) *np.sign(proj_u)/eigv**(1/self.q))
        else:
            raise ValueError('Wrong space!')
        return g
    
    def sample(self):
        """
        Sample a random function u ~ B(0,_C)
        vector u ~ B(0,K): u = gamma |xi|^(1/q), xi ~ EPD(0,1)
        """
        if self.space=='vec':
            epd_rv=gennorm.rvs(beta=self.q,scale=2**(1.0/self.q),size=self.L) # (L,)
            eigv,eigf=self.eigs()
            u=eigv**(1/self.q)*epd_rv # (L,)
        elif self.space=='fun':
            u=super().rnd(n=1).squeeze()
            # u=u.reshape(self.meshsz)
        else:
            raise ValueError('Wrong space!')
        if self.mean is not None:
            u+=self.mean
        return u
    
    def C_act(self,u,comp=1,proj=False):
        """
        Calculate operation of C^comp on vector u: u --> C^comp * u
        """
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
          
        if comp==0:
            return u
        else:
            eigv, eigf=self.eigs()
            if self.space=='vec':
                proj_u=u*eigv**(comp)
            elif self.space=='fun':
                proj_u=eigf.T.dot(u)*eigv**(comp)
            else:
                raise ValueError('Wrong space!')
            if proj or self.space=='vec':
                return proj_u
            else:
                Cu=eigf.dot(proj_u)
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
    
class _qep(qEP):
    """
    q-exponential process prior q-EP(mu,C) defined on 2d domain D.
    """
    def __init__(self,input,L=None,mean=None,store_eig=False,**kwargs):
        x=self._mesh(imsz=input) if np.size(input)<=2 and not isinstance(input,str) else input
        self.space=kwargs.pop('space','vec') # alternative 'fun'
        super().__init__(x=x, L=L, store_eig=store_eig, **kwargs)
        self.mean=mean
    
    def _mesh(self,imsz=None):
        """
        Build the mesh
        """
        if not hasattr(imsz, "__len__"):
            self.imsz=(imsz,)*2
        else:
            self.imsz=imsz
        # set the mesh
        xx,yy=np.meshgrid(np.linspace(0,1,self.imsz[0]),np.linspace(0,1,self.imsz[1]))
        mesh=np.stack([xx.flatten(),yy.flatten()]).T
        # print('\nThe mesh is defined.')
        return mesh
    
    def cost(self,u):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        """
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        u_=u[:,None]
        if self.space=='vec': u_=self.vec2fun(u_)
        val=-self.logpdf(u_)[0]
        return val
    
    def grad(self,u):
        """
        Calculate the gradient of log-prior
        """
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        # r=np.sum(self.C_act(u,comp=-.5)**2)
        r=abs(np.sum(u*self.C_act(u,comp=-1)))
        g=(self.N/2*(1-self.q/2)/r+self.q/4*r**(self.q/2-1))*2*self.C_act(u,comp=-1)
        return g
    
    def sample(self, S=None):
        """
        Sample a random function u ~ qEP(0,_C)
        """
        if S is None:
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
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
          
        if comp==0:
            return u
        else:
            eigv, eigf=self.eigs()
            if comp<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
            Cu=(u if self.space=='vec' else eigf.T.dot(u) if self.space=='fun' else ValueError('Wrong space!'))*eigv**(comp)
            if self.space=='fun': Cu=eigf.dot(Cu)
            return Cu
    
    def normalize(self,u):
        """
        Normalize function u -> r(u)^(-.5)*L^(-1)u, where r(u) = ||u||^2_C and C=LL'
        """
        if self.mean is not None: u-=self.mean
        if np.ndim(u)>1: u=u.flatten()
        
        r=np.sum(self.C_act(u,comp=-.5)**2)
        # r=abs(np.sum(u*self.C_act(u,comp=-1)))
        u_=self.C_act(u,comp=-0.5)/np.sqrt(r)
        return u_
    
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

def prior(prior_option='qep', **kwargs):
    """
    define prior
    """
    return eval('_'+prior_option)(**kwargs)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(2022)
    # define the prior
    prior_option='qep'
    # input=128
    input='https://static.wikia.nocookie.net/nba/images/b/b4/Arizona_State_Sun_Devils.png'
    store_eig=True
    prior = prior(prior_option=prior_option, input=input, basis_opt='Fourier', q=1., L=100, store_eig=store_eig, space='vec', normalize=True)
    # generate sample
    u=prior.sample()
    nlogpri=prior.cost(u)
    ngradpri=prior.grad(u)
    print('The negative logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(nlogpri,np.linalg.norm(ngradpri)))
    # test
    h=1e-5
    v=prior.sample()
    ngradv_fd=(prior.cost(u+h*v)-nlogpri)/h
    ngradv=ngradpri.dot(v.flatten())
    rdiff_gradv=np.abs(ngradv_fd-ngradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    # plot
    if u.shape[0]!=prior.N: u=prior.vec2fun(u)
    plt.imshow(u.reshape(prior.imsz),origin='lower',extent=[0,1,0,1])
    plt.title(prior_option+' sample', fontsize=16)
    plt.show()
    