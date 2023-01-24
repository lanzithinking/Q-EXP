#!/usr/bin/env python
"""
generic q-exponential process
-- with kernel choices 'powered exponential' and 'Matern class'
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ ASU, 2022
-------------------------------
Created August 8, 2022 @ ASU
-------------------------------
https://github.com/lanzithinking
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The Q-EXP project"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# self defined modules
import sys
sys.path.append( "../../" )
from util.kernel.kernel import Ker

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class qEP:
    def __init__(self,x,L=None,store_eig=False,**kwargs):
        """
        Initialize the qEP class with inputs and kernel settings
        x: inputs
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        ker_opt: kernel option, default to be 'powered exponential'
        dist_f: distance function, default to be 'minkowski'
        sigma2: magnitude, default to be 1
        l: correlation length, default to be 0.5
        s: smoothness, default to be 2
        nu: matern class order, default to be 0.5
        q: norm power, default to be 1
        jit: jittering term, default to be 1e-6
        # (dist_f,s)=('mahalanobis',vec) for anisotropic kernel
        spdapx: use speed-up or approximation
        """
        self.x=x # inputs
        self.ker=Ker(x=self.x, L=L, store_eig=store_eig, **kwargs)
        self.q=kwargs.pop('q',1)
    
    def logdet(self):
        """
        Compute log-determinant of the kernel C: log|C|
        """
        eigv,_=self.ker.eigs()
        abs_eigv=abs(eigv)
        ldet=np.sum(np.log(abs_eigv[abs_eigv>=np.finfo(float).eps]))
        return ldet
    
    def logpdf(self,X,chol=False,out='logpdf',incldet=True):
        """
        Compute logpdf of centered q-exponential distribution X ~ qEP(0,C,q)
        """
        assert X.ndim==2 and X.shape[0]==self.ker.N, "Non-conforming size!"
        quad=self.ker.act(X,alpha=-0.5,chol=chol)**2 if chol else X*self.ker.act(X,alpha=-1)
        norms=abs(np.sum(quad,axis=0))**(self.q/2)
        if out=='logpdf':
            half_ldet=-X.shape[1]*self.logdet()/2 if incldet else 0
            quad=-0.5*np.sum(norms)
            log_r=np.log(np.sum(norms))*self.ker.N/2*(1-2/self.q)
            # scal_fctr=X.shape[1]*(np.log(self.ker.q)-self.ker.N/2*np.log(np.pi)-(1+self.N/2)*np.log(2))
            logpdf=half_ldet+quad+log_r#+scal_fctr
            return logpdf,half_ldet#,scal_fctr
        elif out=='norms':
            return norms
    
    def rnd(self,n=1,MU=None):
        """
        Generate q-exponential random function (vector) rv ~ qEP(0,C,q)
        """
        uS_rv=np.random.randn(self.ker.N,n) # (N,n)
        uS_rv/=np.linalg.norm(uS_rv,axis=0,keepdims=True)
        # rv=np.random.gamma(shape=self.ker.N/2,scale=2,size=n)**(1./self.q)*self.ker.act(uS_rv,alpha=0.5)
        rv=np.random.chisquare(df=self.ker.N,size=n)**(1./self.q)*self.ker.act(uS_rv,alpha=0.5)
        if MU is not None:
            rv+=MU
        return rv

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    x=np.linspace(0,1,1000)[:,np.newaxis]
    # x=np.random.rand(100,1) 
    qep=qEP(x,L=100,store_eig=True,cov_opt='matern',s=1.,l=.1,nu=.5,q=1)
    verbose=qep.ker.comm.rank==0 if qep.ker.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(qep.ker.eigv[:min(10,qep.ker.L)],4))
        print('Eigenvectors :', np.round(qep.ker.eigf[:,:min(10,qep.ker.L)],4))
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    v=qep.rnd(n=2)
    C=qep.ker.tomat()
    Cv=C.dot(v)
    Cv_te=qep.ker.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))
    
    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))
    
    v=qep.rnd(n=2)
    solver=spsla.spsolve if sps.issparse(C) else spla.solve
    invCv=solver(C,v)
#     C_op=spsla.LinearOperator((qep.ker.N,)*2,matvec=lambda v:qep.ker.mult(v))
#     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
    invCv_te=qep.ker.act(v,-1)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))
    
#     X=qep.rnd(n=10)
#     X=X.reshape((X.shape[0],5,2),order='F')
#     logpdf,_=qep.logpdf(X)
#     if verbose:
#         print('Log-pdf of an exponential power random variable: {:.4f}'.format(logpdf))
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))
    
    # qep.q=2
    # u=qep.rnd()
    # v=qep.rnd()
    # h=1e-5
    # dlogpdfv_fd=(qep.logpdf(u+h*v)[0]-qep.logpdf(u)[0])/h
    # dlogpdfv=-qep.ker.solve(u).T.dot(v)
    # rdiff_gradv=np.abs(dlogpdfv_fd-dlogpdfv)/np.linalg.norm(v)
    # if verbose:
    #     print('Relative difference of gradients in a random direction between exact calculation and finite difference: %.10f' % rdiff_gradv)
    # if verbose:
    #     print('time: %.5f'% (time.time()-t3))
    
    
    # plot some random samples
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(nrows=1,ncols=2,sharex=False,sharey=False,figsize=(12,5))
    axes[0].plot(qep.ker.eigf[:,:3])
    u=qep.rnd(n=3)
    axes[1].plot(u)
    plt.show()