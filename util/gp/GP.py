#!/usr/bin/env python
"""
generic Gaussian Process
-- with kernel choices 'powered exponential' and 'Matern class'
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ UIUC, 2018
-------------------------------
Created October 26, 2018
-------------------------------
Modified August 15, 2021 @ ASU
-------------------------------
https://bitbucket.org/lanzithinking/tesd_egwas
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2019, TESD project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.9"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import scipy.spatial.distance as spsd
# self defined modules
import sys
sys.path.append( "../../" )
# from util.kernel.covf import *
# from util.kernel.serexp import *
from util.kernel.graphL import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class GP(Ker):
    def __init__(self,x,L=None,store_eig=False,**kwargs):
        """
        Initialize the GP class with inputs and kernel settings
        x: inputs
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        ker_opt: kernel option, default to be 'powered exponential'
        dist_f: distance function, default to be 'minkowski'
        sigma2: magnitude, default to be 1
        l: correlation length, default to be 0.5
        s: smoothness, default to be 2
        nu: matern class order, default to be 0.5
        jit: jittering term, default to be 1e-6
        # (dist_f,s)=('mahalanobis',vec) for anisotropic kernel
        spdapx: use speed-up or approximation
        """
        self.x=x # inputs
        super().__init__(x=self.x, L=L, store_eig=store_eig, **kwargs)
    
    def logdet(self):
        """
        Compute log-determinant of the kernel C: log|C|
        """
        eigv,_=self.eigs()
        abs_eigv=abs(eigv)
        ldet=np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(np.float).eps]))
        return ldet
    
    def logpdf(self,X,nu=1,chol=False):
        """
        Compute logpdf of centered matrix normal distribution X ~ MN(0,C,nu*I)
        """
        assert X.ndim==2 and X.shape[0]==self.N, "Non-conforming size!"
        # if not self.spdapx:
        #     if chol:
        #         try:
        #             cholC,lower=spla.cho_factor(self.tomat())
        #             half_ldet=-X.shape[1]*np.sum(np.log(np.diag(cholC)))
        #             quad=X*spla.cho_solve((cholC,lower),X)
        #         except Exception as e:#spla.LinAlgError:
        #             warnings.warn('Cholesky decomposition failed: '+str(e))
        #             chol=False
        #             pass
        #     if not chol:
        #         half_ldet=-X.shape[1]*self.logdet()/2
        #         quad=X*self.solve(X)
        # else:
        #     eigv,eigf=self.eigs(); rteigv=np.sqrt(abs(eigv)+self.jit)#; rteigv[rteigv<self.jit**2]+=self.jit**2
        #     half_ldet=-X.shape[1]*np.sum(np.log(rteigv))
        #     half_quad=eigf.T.dot(X)/rteigv[:,None]
        #     quad=half_quad**2
        quad=self.act(X,alpha=-0.5,chol=chol)**2 if chol else X*self.act(X,alpha=-1)
        quad=-0.5*np.sum(quad)/nu
        half_ldet=-X.shape[1]*self.logdet()/2
        logpdf=half_ldet+quad
        return logpdf,half_ldet
    
    def rnd(self,n=1,MU=None):
        """
        Generate Gaussian random function (vector) rv ~ N(MU, C)
        """
        mvn0I_rv=np.random.randn(self.N,n) # (N,n)
        rv=self.act(mvn0I_rv,alpha=0.5)
        if MU is not None:
            rv+=MU
        return rv

if __name__=='__main__':
    np.random.seed(2017)
    
    import time
    t0=time.time()
    
    x=np.linspace(0,2*np.pi,100)[:,np.newaxis]
    # x=np.random.randn(100,2)
    gp=GP(x,L=10,store_eig=True,ker_opt='matern',s=1,l=1.,nu=1.5)
    verbose=gp.comm.rank==0 if gp.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(gp.eigv[:min(10,gp.L)],4))
        print('Eigenvectors :', np.round(gp.eigf[:,:min(10,gp.L)],4))
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    v=gp.rnd(n=2)
    C=gp.tomat()
    Cv=C.dot(v)
    Cv_te=gp.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))
    
    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))
    
    v=gp.rnd(n=2)
    solver=spsla.spsolve if sps.issparse(C) else spla.solve
    invCv=solver(C,v)
#     C_op=spsla.LinearOperator((gp.N,)*2,matvec=lambda v:gp.mult(v))
#     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
    invCv_te=gp.act(v,-1)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))
    
#     X=gp.rnd(n=10)
#     X=X.reshape((X.shape[0],5,2),order='F')
#     logpdf,_=gp.logpdf(X)
#     if verbose:
#         print('Log-pdf of a matrix normal random variable: {:.4f}'.format(logpdf))
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))
    
    u=gp.rnd()
    v=gp.rnd()
    h=1e-5
    dlogpdfv_fd=(gp.logpdf(u+h*v)[0]-gp.logpdf(u)[0])/h
    dlogpdfv=-gp.solve(u).T.dot(v)
    rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v)
    if verbose:
        print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
    if verbose:
        print('time: %.5f'% (time.time()-t3))
    
    # plot some random samples
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,figsize=(12,5))
    axes[0].plot(gp.eigf[:,:3])
    u=gp.rnd(n=3)
    axes[1].plot(u)
    plt.show()