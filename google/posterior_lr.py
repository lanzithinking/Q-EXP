#!/usr/bin/env python
"""
Class definition of Gaussian approximation for the posterior measure N(nu,K) with mean function mu and covariance operator K
where K^(-1) = H0(u) + H(u), with H0(u), H(u) being Hessian (or its Gaussian-Newton approximation) of priior and misfit resp.
---- low rank approximation ----
-------------------------------------------------------------------------
Created January 11, 2023 for project of Spatiotemporal Besov prior (STBP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.7"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu lanzithinking@outlook.com"

import os
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# self defined modules
import sys
sys.path.append( "../" )
from util.kernel.linalg import *
from util.Eigen import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class posterior:
    """
    Gaussian approximation of the posterior measure
    """
    def __init__(self,H,H0=None,invH0=None,N=None,store_eig=False,**kwargs):
        self.H = H
        self.H0 = H0
        self.invH0 = invH0
        self.geig = self.H0 is not None and self.invH0 is not None
        self.N = N if N is not None else self.H.shape[0]
        self.spdapx=kwargs.get('spdapx',self.N>1e3)
        self.store_eig=store_eig
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self.eigs(**kwargs)
    
    def mult(self,v,**kwargs):
        """
        Kernel multiply a function (vector): K*v
        """
        return itsol(lambda v:self.H(v)+self.H0(v) if self.geig else self.H,v,solver=kwargs.pop('solver','cgs'))
    
    def solve(self,v):
        """
        Kernel solve a function (vector): K^(-1)*v
        """
        # return self.H.dot(v)
        return self.H(v)+self.H0(v) if self.geig else self.H(v)
    
    def eigs(self,upd=False,**kwargs):
        """
        Obtain partial eigen-basis of H: H * eigf_i = eigf_i * eigv_i, i=1,...,L
        or generalized eigen-basis of pencil (H, H0): H=H0* eigF *eigV *eigF^(-1) such that eigF' *H0 *eigF = I
        """
        if upd or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            k=kwargs.pop('k',100)
            # import time
            if not self.spdapx:
                maxiter=kwargs.pop('maxiter',100)
                tol=kwargs.pop('tol',1e-10)
                try:
                    # start = time.time()
                    H_op=self.H if isinstance(self.H,spsla.LinearOperator) else spsla.LinearOperator((self.N,)*2,self.H)
                    H0_op=self.H0 if self.H0 is None or isinstance(self.H0,spsla.LinearOperator) else spsla.LinearOperator((self.N,)*2,self.H0)
                    invH0_op=self.invH0 if self.invH0 is None or isinstance(self.invH0,spsla.LinearOperator) else spsla.LinearOperator((self.N,)*2,self.invH0)
                    eigv,eigf=spsla.eigs(H_op,k=min(k,self.N-1),which='LM' if self.geig else 'SM',M=H0_op,Minv=invH0_op,maxiter=maxiter,tol=tol)#,which='SM')
                    # end = time.time()
                    # print('Time used is %.4f' % (end-start))
                except Exception as divg:
                    print(*divg.args)
                    eigv,eigf=divg.eigenvalues,divg.eigenvectors
                eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                # eigv=pow(abs(eigv),-1); #eigf=eigf[:,::-1]
                # eigv=np.pad(eigv,(0,k-len(eigv)),mode='constant'); eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
            else:
                p=kwargs.pop('p',10)
                # start = time.time()
                if not self.geig:
                    eigv,eigf=eigen_RA(self.H,dim=self.N,which='SM',k=k,p=p)
                    # eigv=pow(eigv,-1); #eigf=eigf[:,::-1]
                else:
                    eigv,eigf=geigen_RA(self.H,self.H0,self.invH0,dim=self.N,k=k,p=p)
                # end = time.time()
                # print('Time used is %.4f' % (end-start))
                eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
        else:
            eigv,eigf=self.eigv,self.eigf
            # eigv=eigv[:L]; eigf=eigf[:,:L]
        return eigv,eigf
    
    def sample(self, mean=None):
        """
        Sample a random function u ~ N(0, K)
        generalized eigenvalue case:
        K^(1/2)=eigF (I + eigV)^(-1/2)
        """
        # u=self.K_act(np.random.randn(self.N),comp=0.5)
        eigv, eigf=self.eigs()
        u=eigf.dot(np.random.randn(len(eigv))/np.sqrt(eigv+self.geig))
        if mean is not None:
            u+=mean
        return u
    
    def H_act(self,u):
        """
        Calculate low-rank approximation of H
        """
        if not self.spdapx:
            return self.H(u)
        else:
            eigv, eigf=self.eigs()
            if not self.geig:
                return np.tensordot(eigf*eigv,np.tensordot(eigf,u,axes=(0,0)),axes=1)
            else:
                return self.H0(np.tensordot(eigf*eigv,np.tensordot(eigf,self.H0(u),axes=(0,0)),axes=1))
    
    def K_act(self,u,comp=1,**kwargs):
        """
        Calculate operation of K^comp on vector u: u --> K^comp * u
        generalized eigenvalue case:
        K^(-1) = H_0 + H = H_0 + H_0 eigF eigV eigF' H_0 ~ H_0 eigF (I+ eigV) eigF' H_0
        K = [H_0 + H]^(-1) = H_0^(-1) - eigF (eigV^(-1) + I)^(-1) eigF' ~ eigF (I + eigV)^(-1) eigF'
        """
        if comp==0:
            return u
        else:
            if not self.spdapx:
                Ku=self.mult(u,**kwargs) if comp==1 else self.solve(u) if comp==-1 else None
            else:
                eigv, eigf=self.eigs()
                if comp==1:
                    if not self.geig:
                        Ku=np.tensordot(eigf/eigv,np.tensordot(eigf,u,axes=(0,0)),axes=1)
                    else:
                        # Ku=self.invH0(u)-np.tensordot(eigf*eigv/(eigv+1),np.tensordot(eigf,u,axes=(0,0)),axes=1) 
                        Ku=np.tensordot(eigf/(eigv+1),np.tensordot(eigf,u,axes=(0,0)),axes=1)
                elif comp==-1:
                    if not self.geig:
                        Ku=np.tensordot(eigf*eigv,np.tensordot(eigf,u,axes=(0,0)),axes=1)
                    else:
                        # H0u=self.H0(u)
                        # Ku=H0u+ self.H0(np.tensordot(eigf*eigv,np.tensordot(eigf,H0u,axes=(0,0)),axes=1))
                        Ku=self.H0(np.tensordot(eigf*(eigv+1),np.tensordot(eigf,self.H0(u),axes=(0,0)),axes=1))
                else:
                    raise NotImplementedError('Action not defined!')
        return Ku
    
    def logdet(self,eigv=None):
        """
        Compute log-determinant of the kernel K: log|K|
        generalized eigenvalue case: log|K|=-log|I+eigV|+log|H_0|
        """
        if eigv is None:
            eigv,_=self.eigs()
        abs_eigv=abs(eigv+self.geig)
        ldet=-np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(float).eps]))
        return ldet
    
# if __name__ == '__main__':
    # np.random.seed(2022)
    