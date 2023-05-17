'''
Prior of the Advection-Diffusion problem written in FEniCS-2019.1.0 and hIPPYlib-3.0
https://hippylib.github.io/tutorials_v3.0.0/4_AdvectionDiffusionBayesian/
-------------------------------------------------------------------------
Project of Bayesian SpatioTemporal analysis for Inverse Problems (B-STIP)
Shiwei Lan @ ASU, Sept. 2020
--------------------------------------------------------------------------
Created on Sep 23, 2020
-------------------------------
https://github.com/lanzithinking/Spatiotemporal-inverse-problem
'''
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2020, The Bayesian STIP project"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
# sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
# priors
from _GP import _GP
from _Besov import _Besov
from _qEP import _qEP

import warnings
warnings.simplefilter('once')

def _prior(prior_option='qep', **kwargs):
    """
    define prior
    """
    return eval('_'+{'gp':'GP','bsv':'Besov','qep':'qEP'}[prior_option])(**kwargs)
    
if __name__=='__main__':
    from pde import *
    np.random.seed(2020)
    # obtain function space
#     mesh = dl.Mesh('ad_10k.xml')
#     Vh = dl.FunctionSpace(mesh, "Lagrange", 2)
    meshsz = (61,61)
    eldeg = 1
    pde = TimeDependentAD(mesh=meshsz, eldeg=eldeg)
    Vh = pde.Vh[STATE]
    # define prior
    prior_option='qep'
    gamma = 2.; delta = 10.; L = 1000
    prior = _prior(prior_option=prior_option, Vh=Vh, gamma=gamma, delta=delta, L=L)
    
    # test gradient with finite difference
    u=prior.sample()
    l,g=prior.logpdf(u,add_mean=True,grad=True)
    h=1e-6
    v=prior.sample()
    Hv=prior.gen_vector()
    prior.hess(u, v, Hv)
    u.axpy(h,v)
    l1,g1=prior.logpdf(u,add_mean=True,grad=True)
    dlv_fd = (l1-l)/h
    dlv = g.inner(v)
    rdiff_gradv = np.abs(dlv_fd-dlv)/v.norm('l2')
    print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    Hv_fd=(g1-g)/h
    rdiff_hessv=(Hv_fd+Hv).norm('l2')/v.norm('l2')
    print('Relative difference of Hessian-action in a random direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
    
    # check conversions
    whiten=False
    u=prior.sample(whiten=whiten)
    logpri,gradpri=prior.logpdf(u, whiten=whiten, grad=True)
    print('The logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri,gradpri.norm('l2')))
    fig=dl.plot(vector2Function(u,prior.Vh))
    plt.colorbar(fig); plt.show()
    v=prior.u2v(u)
    logpri_wt,gradpri_wt=prior.logpdf(v, whiten=True, grad=True)
    print('The logarithm of prior density at whitened v is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri_wt,gradpri_wt.norm('l2')))
    fig=dl.plot(vector2Function(v,prior.Vh))
    plt.colorbar(fig); plt.show()
    
    whiten=True
#     v=prior.sample(whiten=whiten)
    logpri,gradpri=prior.logpdf(v, whiten=whiten, grad=True)
    print('The logarithm of prior density at whitened v is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri,gradpri.norm('l2')))
    fig=dl.plot(vector2Function(v,prior.Vh))
    plt.colorbar(fig); plt.show()
    u=prior.v2u(v)
    logpri_wt,gradpri_wt=prior.logpdf(u, whiten=False, grad=True)
    print('The logarithm of prior density at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(logpri_wt,gradpri_wt.norm('l2')))
    fig=dl.plot(vector2Function(u,prior.Vh))
    plt.colorbar(fig); plt.show()