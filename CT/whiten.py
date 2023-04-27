#!/usr/bin/env python
"""
Class definition of whitening non-Gaussian (STBP) distribution
-------------------------------------------------------------------------
Created February 1, 2023 for project of Spatiotemporal Besov prior (STBP)
updated April 23, 2023 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project and the Q-EXP project"
__license__ = "GPL"
__version__ = "0.4"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu lanzithinking@outlook.com"

import os
import numpy as np

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class whiten:
    """
    Whiten the non-Gaussian (Q-EP) prior
    """
    def __init__(self,prior):
        self.prior = prior # qep prior
        self.mean = None if self.prior.mean is None else self.qep2wn(self.prior.mean)
    
    def wn2qep(self,z,dord=0,q=None):
        """
        White noise (z) representation of a q-EP random variable (xi), Lmd: z --> xi
        """
        if q is None: q = self.prior.q
        _z = z.reshape((self.prior.dim,-1),order='F') # dim = L (vec) or N (fun)
        nm_z = np.linalg.norm(_z,axis=0,keepdims=True)
        if dord==0:
            return self.prior.C_act(_z*nm_z**(2/q-1),comp=0.5).squeeze()# (dim,_)
        if dord==1:
            def grad(v, adj=False):
                _v = v.reshape((self.prior.dim,-1),order='F')
                if adj:
                    _v = self.prior.C_act(_v, comp=0.5)
                    dLmdv = _z*np.sum(_z*_v,axis=0,keepdims=True)*nm_z**(2/q-3)*(2/q-1) + _v*nm_z**(2/q-1)
                    return dLmdv.squeeze()
                else:
                    return self.prior.C_act(_z*np.sum(_z*_v,axis=0,keepdims=True)*nm_z**(2/q-3)*(2/q-1) + _v*nm_z**(2/q-1), comp=0.5).squeeze()#,chol=False)
            return grad
        if dord==2:
            def hess(v, w, adj=False):
                _v = v.reshape((self.prior.dim,-1),order='F')
                _w = w.reshape((self.prior.dim,-1),order='F')
                Hv0 = (2/q-1)*self.prior.C_act(_w*np.sum(_z*_v,axis=0,keepdims=True)*nm_z**(2/q-3), comp=0.5)
                Hv1 = (2/q-1)*self.prior.C_act(_z*nm_z**(2/q-3), comp=0.5)
                Hv2 = (2/q-1)*self.prior.C_act(_z*np.sum(_z*_v,axis=0,keepdims=True)*nm_z**(2/q-5)*(2/q-3) + _v*nm_z**(2/q-3), comp=0.5)
                if adj:
                    wHv = Hv0 + np.sum(_w*Hv1,axis=0,keepdims=True)*_v + np.sum(_w*Hv2,axis=0,keepdims=True)*_z
                else:
                    wHv = Hv0 + np.sum(_w*_v,axis=0,keepdims=True)*Hv1 + np.sum(_w*_z,axis=0,keepdims=True)*Hv2
                return wHv.squeeze()
            return hess
    
    def qep2wn(self,xi,dord=0,q=None):
        """
        Inverse white noise (z) representation of a q-EP random variable (xi), invLmd: xi --> z
        """
        if q is None: q = self.prior.q
        _xi = self.prior.C_act(xi.reshape((self.prior.dim,-1),order='F'),comp=-0.5) # (dim,_)
        nm_xi = np.linalg.norm(_xi,axis=0,keepdims=True)
        if dord==0:
            return (_xi*nm_xi**(q/2-1)).squeeze()
        if dord==1:
            def grad(v, adj=False):
                _v = v.reshape((self.prior.dim,-1),order='F')
                if adj:
                    return self.prior.C_act(_xi*np.sum(_xi*_v,axis=0,keepdims=True)*nm_xi**(q/2-3)*(q/2-1) + _v*nm_xi**(q/2-1), comp=-0.5).squeeze()
                else:
                    _v = self.prior.C_act(_v, comp=-0.5)
                    diLmdv = _xi*np.sum(_xi*_v,axis=0,keepdims=True)*nm_xi**(q/2-3)*(q/2-1) + _v*nm_xi**(q/2-1)
                    return diLmdv.squeeze()
            return grad
    
    def jacdet(self,z,dord=0,q=None):
        """
        (log) Jacobian determinant log dLmd
        """
        if q is None: q = self.prior.q
        _z = z.reshape((self.prior.dim,-1),order='F') # (dim,_)
        nm_z = np.linalg.norm(_z,axis=0,keepdims=True)
        if dord==0:
            return (2/q-1)*self.prior.dim*np.log(nm_z).sum()
        if dord==1:
            return (2/q-1)*self.prior.dim*(_z/nm_z**2).reshape(z.shape,order='F')
    
    def sample(self, mean=None):
        """
        Generate white noise sample
        """
        if mean is None:
            mean=self.mean
        u=np.random.randn(self.prior.dim)
        if mean is not None:
            u+=mean
        return u
    
if __name__ == '__main__':
    from prior import *
    
    seed=2022
    np.random.seed(seed)
    # define the prior
    prior_option='qep'
    # input=128
    input='satellite.png'
    store_eig=True
    prior = prior(prior_option=prior_option, ker_opt='graphL', input=input, basis_opt='Fourier', q=1.01, L=200, store_eig=store_eig, space='vec', normalize=True, weightedge=True)
    prior.mean = prior.sample()[:,None]
    # define whitened object
    wht = whiten(prior)
    
    # test
    h=1e-8; z, v, w=np.random.randn(3,prior.dim)
    # wn2qep (Lmd)
    print('**** Testing wn2qep (Lmd) ****')
    val,grad,hess=wht.wn2qep(z,0),wht.wn2qep(z,1),wht.wn2qep(z,2)
    val1,grad1=wht.wn2qep(z+h*v,0),wht.wn2qep(z+h*w,1)
    print('error in gradient: %0.8f' %(np.linalg.norm((val1-val)/h-grad(v))/np.linalg.norm(v)))
    print('error in Hessian: %0.8f' %(np.linalg.norm((grad1(v)-grad(v))/h-hess(v,w))/np.sqrt(np.linalg.norm(v)*np.linalg.norm(w))))
    
    h=1e-8; xi, v=np.random.randn(2,prior.dim)
    # qep2wn (invLmd)
    print('\n**** Testing qep2wn (invLmd) ****')
    val,grad=wht.qep2wn(xi,0),wht.qep2wn(xi,1)
    val1=wht.qep2wn(xi+h*v,0)
    print('error in gradient: %0.8f' %(np.linalg.norm((val1-val)/h-grad(v))/np.linalg.norm(v)))
    xi1=wht.wn2qep(val,0).flatten(order='F')
    print('Relative error of Lmd-invLmd in a random direction between composition and identity: %.10f' % (np.linalg.norm(xi1-xi)/np.linalg.norm(xi)) )
    xi2=wht.qep2wn(wht.wn2qep(xi,0),0).flatten(order='F')
    print('Relative error of invLmd-Lmd in a random direction between composition and identity: %.10f' % (np.linalg.norm(xi2-xi)/np.linalg.norm(xi)))
    gradv=grad(v)
    v1=wht.wn2qep(val,1)(gradv).flatten(order='F')
    print('Relative error of dLmd-dinvLmd in a random direction between composition and identity: %.10f' % (np.linalg.norm(v1-v)/np.linalg.norm(v)))
    v2=wht.qep2wn(xi,1)(wht.wn2qep(val,1)(v)).flatten(order='F')
    print('Relative error of dinvLmd-dLmd in a random direction between composition and identity: %.10f' % (np.linalg.norm(v2-v)/np.linalg.norm(v)))
    
    h=1e-8; z, v=np.random.randn(2,prior.dim)
    # jacdet
    print('\n**** Testing jacdet ****')
    val,grad=wht.jacdet(z,0),wht.jacdet(z,1)
    val1=wht.jacdet(z+h*v,0)
    print('error in gradient of Jacobian determinant: %0.8f' %(np.linalg.norm((val1-val)/h-grad.dot(v))/np.linalg.norm(v)))