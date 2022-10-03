#!/usr/bin/env python
"""
Kernel class
-- with kernel options 'covf'(covariance function0, 'serexp'(series expansion) and 'graphL'(graph Laplacian)
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ ASU, 2022
----------------------
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.1"
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
from util.kernel.covf import Ker as Ker_covf
from util.kernel.serexp import Ker as Ker_serexp
from util.kernel.graphL import Ker as Ker_graphL

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def Ker(ker_opt='serexp', **kwargs):
    """
    define the kernel
    """
    ker=eval('Ker_'+ker_opt)(**kwargs)
    ker.opt=ker_opt
    return ker

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    #     x=np.linspace(0,2*np.pi)[:,np.newaxis]
    x=np.random.randn(100,2)
    ker=Ker(x=x,L=10,store_eig=True,cov_opt='matern')
    verbose=ker.comm.rank==0 if ker.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(ker.eigv[:min(10,ker.L)],4))
        print('Eigenvectors :', np.round(ker.eigf[:,:min(10,ker.L)],4))
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    v=x
    C=ker.tomat()
    Cv=C.dot(v)
    Cv_te=ker.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))
    
    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))
    
    invCv=np.linalg.solve(C,v)
#     C_op=spsla.LinearOperator((ker.N,)*2,matvec=lambda v:ker.mult(v))
#     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
    invCv_te=ker.act(v,-1)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))
    
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))