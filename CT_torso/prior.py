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
__version__ = "0.8"
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
        if self.ker.opt=='graphL': self.imsz=self.ker.imsz
        self.dim={'vec':self.ker.L,'fun':self.ker.N}[self.space]
        self.mean=mean
        if self.mean is not None:
            assert self.mean.size==self.dim, "Non-conforming size of mean!"
    
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
    
    def cost(self,u,**kwargs):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        if self.space=='vec': u=self.vec2fun(u)
        val=-self.logpdf(u,incldet=kwargs.pop('incldet',True))[0]
        return np.squeeze(val)
    
    def grad(self,u):
        """
        Calculate the gradient of log-prior
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        g=self.C_act(u,comp=-1)
        return g.squeeze()
    
    def Hess(self,u):
        """
        Calculate the Hessian action of log-prior
        """
        # if u.ndim==1 or u.shape[0]!=self.dim:
        #     u=u.reshape((self.dim,-1),order='F')
        # if self.mean is not None: u=u-self.mean
        #
        # invCu=self.C_act(u,comp=-1)
        # r=abs(np.sum(u*invCu))
        def hess(v):
            if v.ndim==1 or v.shape[0]!=self.dim:
                v=v.reshape((self.dim,-1),order='F')
            Hv=self.C_act(v,comp=-1)
            return Hv.squeeze()
        return hess
    
    def invHess(self,u):
        """
        Calculate the inverse Hessian action of log-prior
        """
        # if u.ndim==1 or u.shape[0]!=self.dim:
        #     u=u.reshape((self.dim,-1),order='F')
        # if self.mean is not None: u=u-self.mean
        #
        # invCu=self.C_act(u,comp=-1)
        # r=abs(np.sum(u*invCu))
        def ihess(v):
            if v.ndim==1 or v.shape[0]!=self.dim:
                v=v.reshape((self.dim,-1),order='F')
            iHv=self.C_act(v)
            return iHv.squeeze()
        return ihess
    
    def sample(self,**kwargs):
        """
        Sample a random function u ~ N(0,_C)
        """
        Z=np.random.randn(self.dim)
        u=self.C_act(Z,comp=0.5,**kwargs)
        if self.mean is not None:
            u+=self.mean
        return u.squeeze()
    
    def C_act(self,u,comp=1,**kwargs):
        """
        Calculate operation of C^comp on vector u: u --> C^comp * u
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        # if self.mean is not None: u=u-self.mean
          
        if comp==0:
            return u
        else:
            if self.space=='vec':
                eigv, eigf=self.ker.eigs()
                if comp<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                Cu=u*eigv[:,None]**(comp)
            elif self.space=='fun':
                Cu=self.ker.act(u,alpha=comp,**kwargs)
            return Cu
    
    def vec2fun(self, u_vec):
        """
        Convert vector (u_i) to function (u)
        """
        eigv, eigf = self.ker.eigs()
        u_f = eigf.dot(u_vec)
        return u_f
    
    def fun2vec(self, u_f):
        """
        Convert vector (u_i) to function (u)
        """
        eigv, eigf = self.ker.eigs()
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
        if self.ker.opt=='graphL': self.imsz=self.ker.imsz
        self.dim={'vec':self.ker.L,'fun':self.ker.N}[self.space]
        self.mean=mean
        if self.mean is not None:
            assert self.mean.size==self.dim, "Non-conforming size of mean!"
    
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
    
    def cost(self,u,**kwargs):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        -.5* ||_C^(-1/q) u(x)||^q = -.5 sum_l |gamma_l^{-1} <phi_l, u>|^q
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        proj_u=self.C_act(u, -1.0/self.q, proj=True)
        val=0.5*np.sum(abs(proj_u)**self.q)
        return np.squeeze(val)
    
    def grad(self,u):
        """
        Calculate the gradient of log-prior
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        eigv, eigf=self.ker.eigs()
        if self.space=='vec':
            # proj_u=u/eigv[:,None]**(1/self.q)
            # g=0.5*self.q*abs(proj_u)**(self.q-1) *np.sign(proj_u)/eigv[:,None]**(1/self.q)
            g=0.5*self.q*abs(u)**(self.q-1) *np.sign(u)/eigv[:,None]
        elif self.space=='fun':
            # proj_u=eigf.T.dot(u)/eigv[:,None]**(1/self.q)
            # g=eigf.dot(0.5*self.q*abs(proj_u)**(self.q-1) *np.sign(proj_u)/eigv[:,None]**(1/self.q))
            proj_u=eigf.T.dot(u)
            g=eigf.dot(0.5*self.q*abs(proj_u)**(self.q-1) *np.sign(proj_u)/eigv[:,None])
        else:
            raise ValueError('Wrong space!')
        return g.squeeze()
    
    def Hess(self,u):
        """
        Calculate the Hessian action of log-prior
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        eigv, eigf=self.ker.eigs()
        def hess(v):
            if v.ndim==1 or v.shape[0]!=self.dim:
                v=v.reshape((self.dim,-1),order='F')
            if self.space=='vec':
                # proj_u=u/eigv[:,None]**(1/self.q)
                # Hv=0.5*self.q*(self.q-1)* (abs(proj_u)**(self.q-2)/eigv[:,None]**(2/self.q))[:,None]*v
                Hv=0.5*self.q*(self.q-1)* (abs(u)**(self.q-2)/eigv[:,None])*v
            elif self.space=='fun':
                # proj_u=eigf.T.dot(u)/eigv[:,None]**(1/self.q)
                # Hv=eigf.dot(0.5*self.q*(self.q-1)* (abs(proj_u)**(self.q-2)/eigv[:,None]**(2/self.q))[:,None]*eigf.T.dot(v))
                proj_u=eigf.T.dot(u)
                Hv=eigf.dot(0.5*self.q*(self.q-1)* (abs(proj_u)**(self.q-2)/eigv[:,None])*eigf.T.dot(v))
            else:
                raise ValueError('Wrong space!')
            return Hv.squeeze()
        return hess
    
    def invHess(self,u):
        """
        Calculate the inverse Hessian action of log-prior
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        eigv, eigf=self.ker.eigs()
        def ihess(v):
            if v.ndim==1 or v.shape[0]!=self.dim:
                v=v.reshape((self.dim,-1),order='F')
            if self.space=='vec':
                iHv=(2/self.q/(self.q-1) if self.q!=1 else 0)* (abs(u)**(-self.q+2)*eigv[:,None])*v
            elif self.space=='fun':
                proj_u=eigf.T.dot(u)
                iHv=eigf.dot((2/self.q/(self.q-1) if self.q!=1 else 0)* (abs(proj_u)**(-self.q+2)*eigv[:,None])*eigf.T.dot(v))
            else:
                raise ValueError('Wrong space!')
            return iHv.squeeze()
        return ihess
    
    def sample(self,**kwargs):
        """
        Sample a random function u ~ B(0,_C)
        vector u ~ B(0,K): u = gamma |xi|^(1/q), xi ~ EPD(0,1)
        """
        if self.space=='vec':
            epd_rv=gennorm.rvs(beta=self.q,scale=2**(1.0/self.q),size=self.ker.L) # (L,)
            eigv,eigf=self.ker.eigs()
            u=eigv**(1/self.q)*epd_rv # (L,)
        elif self.space=='fun':
            u=super().rnd(n=1).squeeze()
            # u=u.reshape(self.imsz)
        else:
            raise ValueError('Wrong space!')
        if self.mean is not None:
            u+=self.mean
        return u.squeeze()
    
    def C_act(self,u,comp=1,**kwargs):
        """
        Calculate operation of C^comp on vector u: u --> C^comp * u
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        # if self.mean is not None: u=u-self.mean
          
        if comp==0:
            return u
        else:
            eigv, eigf=self.ker.eigs()
            if comp<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
            if self.space=='vec':
                proj_u=u*eigv[:,None]**(comp)
            elif self.space=='fun':
                proj_u=eigf.T.dot(u)*eigv[:,None]**(comp)
            else:
                raise ValueError('Wrong space!')
            if kwargs.pop('proj',False) or self.space=='vec':
                return proj_u
            else:
                Cu=eigf.dot(proj_u)
                return Cu
    
    def vec2fun(self, u_vec):
        """
        Convert vector (u_i) to function (u)
        """
        eigv, eigf = self.ker.eigs()
        u_f = eigf.dot(u_vec)
        return u_f
    
    def fun2vec(self, u_f):
        """
        Convert vector (u_i) to function (u)
        """
        eigv, eigf = self.ker.eigs()
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
        if self.ker.opt=='graphL': self.imsz=self.ker.imsz
        self.dim={'vec':self.ker.L,'fun':self.ker.N}[self.space]
        self.mean=mean
        if self.mean is not None:
            assert self.mean.size==self.dim, "Non-conforming size of mean!"
    
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
    
    def cost(self,u,**kwargs):
        """
        Calculate the logarithm of prior density (and its gradient) of function u.
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        # if self.space=='vec': u=self.vec2fun(u)
        # val=-self.logpdf(u,incldet=kwargs.pop('incldet',True))[0]
        invCu=self.C_act(u,comp=-1)
        r=abs(np.sum(u*invCu,axis=0,keepdims=True))
        val=self.ker.N/2*(1-self.q/2)*np.log(r)+r**(self.q/2)/2
        return np.squeeze(val)
    
    def grad(self,u):
        """
        Calculate the gradient of log-prior
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        invCu=self.C_act(u,comp=-1)
        r=abs(np.sum(u*invCu,axis=0,keepdims=True))
        A=(self.ker.N*(1-self.q/2)+self.q/2*r**(self.q/2))/r
        g=A*invCu
        return g.squeeze()
    
    def Hess(self,u,logr=True):
        """
        Calculate the Hessian action of log-prior
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        invCu=self.C_act(u,comp=-1)
        r=abs(np.sum(u*invCu,axis=0,keepdims=True))
        A=(self.ker.N*(1-self.q/2)*logr+self.q/2*r**(self.q/2))/r
        B=(self.q-2)*(self.ker.N*logr+self.q/2*r**(self.q/2))/r**2
        def hess(v):
            if v.ndim==1 or v.shape[0]!=self.dim:
                v=v.reshape((self.dim,-1),order='F')
            invCv=self.C_act(v,comp=-1)
            Hv=A*invCv+B*invCu*np.sum(u*invCv,axis=0,keepdims=True)
            return Hv.squeeze()
        return hess
    
    def invHess(self,u,logr=True):
        """
        Calculate the inverse Hessian action of log-prior
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        invCu=self.C_act(u,comp=-1)
        r=abs(np.sum(u*invCu,axis=0,keepdims=True))
        A=(self.ker.N*(1-self.q/2)*logr+self.q/2*r**(self.q/2))/r
        B=(self.q-2)*(self.ker.N*logr+self.q/2*r**(self.q/2))/r**2
        C=A+B*r
        def ihess(v):
            if v.ndim==1 or v.shape[0]!=self.dim:
                v=v.reshape((self.dim,-1),order='F')
            iHv=self.C_act(v)
            if np.any(C): iHv+=-B/C*u*np.sum(u*v,axis=0,keepdims=True)
            iHv/=A
            return iHv.squeeze()
        return ihess
    
    def sample(self,S=None,**kwargs):
        """
        Sample a random function u ~ qEP(0,_C)
        """
        if S is None:
            S=np.random.randn(self.dim)
            S/=np.linalg.norm(S)
        R=np.random.chisquare(df=self.ker.N)**(1./self.q)
        u=R*self.C_act(S,comp=0.5,**kwargs)
        if self.mean is not None:
            u+=self.mean
        return u.squeeze()
    
    def C_act(self,u,comp=1,**kwargs):
        """
        Calculate operation of C^comp on vector u: u --> C^comp * u
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        # if self.mean is not None: u=u-self.mean
        
        if comp==0:
            return u
        else:
            if self.space=='vec':
                eigv, eigf=self.ker.eigs()
                if comp<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                Cu=u*eigv[:,None]**(comp)
            elif self.space=='fun':
                Cu=self.ker.act(u,alpha=comp,**kwargs)
            return Cu
    
    def normalize(self,u):
        """
        Normalize function u -> r(u)^(-.5)*L^(-1)u, where r(u) = ||u||^2_C and C=LL'
        """
        if u.ndim==1 or u.shape[0]!=self.dim:
            u=u.reshape((self.dim,-1),order='F')
        if self.mean is not None: u=u-self.mean
        
        r=np.sum(self.C_act(u,comp=-.5)**2)
        # r=abs(np.sum(u*self.C_act(u,comp=-1)))
        u_=self.C_act(u,comp=-0.5)/np.sqrt(r)
        return u_
    
    def vec2fun(self, u_vec):
        """
        Convert vector (u_i) to function (u)
        """
        eigv, eigf = self.ker.eigs()
        u_f = eigf.dot(u_vec)
        return u_f
    
    def fun2vec(self, u_f):
        """
        Convert vector (u_i) to function (u)
        """
        eigv, eigf = self.ker.eigs()
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
    input='satellite.png'
    store_eig=True
    prior = prior(prior_option=prior_option, ker_opt='graphL', input=input, basis_opt='Fourier', q=1.01, L=200, store_eig=store_eig, space='fun', normalize=True, weightedge=True)
    # generate sample
    u=prior.sample()
    nlogpri=prior.cost(u)
    ngradpri=prior.grad(u)
    print('The negative logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(nlogpri,np.linalg.norm(ngradpri)))
    hess=prior.Hess(u)
    # test
    h=1e-5
    v=prior.sample()
    ngradv_fd=(prior.cost(u+h*v)-nlogpri)/h
    ngradv=ngradpri.dot(v.flatten())
    rdiff_gradv=np.abs(ngradv_fd-ngradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    hessv_fd=(prior.grad(u+h*v)-ngradpri)/h
    hessv=hess(v)
    rdiff_hessv=np.linalg.norm(hessv_fd-hessv)/np.linalg.norm(v)
    print('Relative difference of Hessian-action in a random direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
    ihess=prior.invHess(u)
    v1=ihess(hessv)
    rdiff_v1=np.linalg.norm(v1-v.flatten())/np.linalg.norm(v)
    print('Relative difference of invHessian-Hessian-action in a random direction between the composition and identity: %.10f' % rdiff_v1)
    v2=hess(ihess(v))
    rdiff_v2=np.linalg.norm(v2-v.flatten())/np.linalg.norm(v)
    print('Relative difference of Hessian-invHessian-action in a random direction between the composition and identity: %.10f' % rdiff_v2)
    # plot
    plt.rcParams['image.cmap'] = 'binary'
    if u.shape[0]!=prior.ker.N: u=prior.vec2fun(u)
    plt.imshow(u.reshape(prior.imsz),origin='lower',extent=[0,1,0,1])
    plt.title(prior_option+' sample', fontsize=16)
    plt.show()
    