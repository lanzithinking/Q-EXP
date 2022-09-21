#!/usr/bin/env python
"""
generic Besov measure
-- with basis choices 'wavelet' and 'Fourier'
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ ASU, 2022
-------------------------------
Created February 10, 2022 @ ASU
-------------------------------
https://github.com/lanzithinking
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The Q-EXP project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.6"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.stats import gennorm
# self defined modules
import sys
sys.path.append( "../../" )
# from util.kernel.covf import *
from util.kernel.serexp import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class BSV(Ker):
    def __init__(self,x,L=None,store_eig=False,**kwargs):
        """
        Initialize the Besov class with inputs and kernel settings
        x: inputs
        basis_opt: basis option, default to be 'Fourier'
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        sigma2: magnitude, default to be 1
        l: inverse correlation length, default to be 0.5
        s: smoothness, default to be 2
        q: norm power, default to be 1
        jit: jittering term, default to be 1e-6
        spdapx: use speed-up or approximation
        """
        self.x=x # inputs
        if self.x.ndim==1: self.x=self.x[:,None]
        super().__init__(x=self.x, L=L, store_eig=store_eig, **kwargs)
        if not hasattr(self,'q'): self.q=kwargs.pop('q',1.0)
    
    def logdet(self):
        """
        Compute log-determinant of the kernel C: log|C|
        """
        eigv,_=self.eigs()
        abs_eigv=abs(eigv)
        ldet=np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(float).eps]))
        return ldet
    
    def logpdf(self,X):
        """
        Compute logpdf of centered Besov distribution X ~ Besov(0,C)
        """
        eigv,eigf=self.eigs()
        if not self.spdapx:
            proj_X = eigf.T.dot(self.act(X, alpha=-1/self.q))
            q_ldet=-X.shape[1]*self.logdet()/self.q
        else:
            qrt_eigv=eigv**(1/self.q)
            q_ldet=-X.shape[1]*np.sum(np.log(qrt_eigv))
            proj_X=eigf.T.dot(X)/qrt_eigv[:,None]
        qsum=-0.5*np.sum(abs(proj_X)**self.q)
        logpdf=q_ldet+qsum
        return logpdf,q_ldet
    
    def update(self,sigma2=None,l=None):
        """
        Update the eigen-basis
        """
        if sigma2 is not None:
            sigma2_=self.sigma2
            self.sigma2=sigma2
            if self.store_eig:
                self.eigv*=self.sigma2/sigma2_
        if l is not None:
            self.l=l
            if self.store_eig:
                self.eigv,self.eigf=self.eigs(upd=True)
        return self
    
    def rnd(self,n=1):
        """
        Generate Besov random function (vector) rv ~ Besov(0,C)
        """
        epd_rv=gennorm.rvs(beta=self.q,scale=2**(1.0/self.q),size=(self.L,n)) # (L,n)
        eigv,eigf=self.eigs()
        rv=eigf.dot(eigv[:,None]**(1/self.q)*epd_rv) # (N,n)
        return rv

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    meshsz=32
    # x=np.random.rand(meshsz**2,2)
    # x=np.stack([np.sort(np.random.rand(meshsz**2)),np.sort(np.random.rand(meshsz**2))]).T
    xx,yy=np.meshgrid(np.linspace(0,1,meshsz),np.linspace(0,1,meshsz))
    x=np.stack([xx.flatten(),yy.flatten()]).T
    bsv=BSV(x,L=100,store_eig=True,basis_opt='Fourier', l=.1, q=1.0, spdapx=True) # constrast with q=2.0
    verbose=bsv.comm.rank==0 if bsv.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(bsv.eigv[:min(10,bsv.L)],4))
        print('Eigenvectors :', np.round(bsv.eigf[:,:min(10,bsv.L)],4))

    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))

    u_samp=bsv.rnd(n=25)
    v=bsv.rnd(n=2)
    C=bsv.tomat()
    Cv=C.dot(v)
    Cv_te=bsv.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))

    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))

    v=bsv.rnd(n=2)
    invCv=np.linalg.solve(C,v)
#     C_op=spsla.LinearOperator((bsv.N,)*2,matvec=lambda v:bsv.mult(v))
#     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
    invCv_te=bsv.act(v,-1)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))

    X=bsv.rnd(n=10)
    logpdf,_=bsv.logpdf(X)
    if verbose:
        print('Log-pdf of a matrix normal random variable: {:.4f}'.format(logpdf))
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))

    # bsv.q=2;
    # u=bsv.rnd()
    # v=bsv.rnd()
    # h=1e-6
    # dlogpdfv_fd=(bsv.logpdf(u+h*v)[0]-bsv.logpdf(u)[0])/h
    # dlogpdfv=-bsv.solve(u).T.dot(v)
    # rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v)
    # if verbose:
    #     print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
    # if verbose:
    #     print('time: %.5f'% (time.time()-t3))
    
    # plot some random samples
    import matplotlib.pyplot as plt
    # fig, axes=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True,figsize=(15,12))
    # for i,ax in enumerate(axes.flat):
    #     ax.imshow(bsv.eigf[:,i].reshape((int(np.sqrt(bsv.eigf.shape[0])),-1)),origin='lower')
    #     ax.set_aspect('auto')
    # plt.show()
    
    fig, axes=plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True,figsize=(15,12))
    for i,ax in enumerate(axes.flat):
        ax.imshow(u_samp[:,i].reshape((int(np.sqrt(u_samp.shape[0])),-1)),origin='lower')
        ax.set_aspect('auto')
    plt.show()