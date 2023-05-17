#!/usr/bin/env python
"""
Class definition of whitening Gaussian prior
Given N(mu,C), output N(0,I)
---------------------------------------------------------------
written in FEniCS 2017.1.0-dev, with backward support for 1.6.0
Shiwei Lan @ Caltech, Sept. 2017
---------------------------------------------------------------
Created March 28, 2017
---------------------------------------------------------------
Modified in May, 2023 in FEniCS 2019.1.0 (python 3) @ ASU
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2017, The EQUiPS project"
__license__ = "GPL"
__version__ = "0.4"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@caltech.edu; lanzithinking@outlook.com; slan@asu.edu"

import dolfin as dl
import numpy as np
from hippylib import *

class wht_prior:
    """
    Whiten the non-Gaussian (Q-EP) prior q-EP(mu,C), assembled(C^(-1)) = R = A * M^(-1) * A
    """
    def __init__(self,prior):
        self.prior=prior
        self.R=prior.M # for the convenience of defining GaussianLRPosterior
        self.Rsolver=prior.Msolver
    
    def init_vector(self,x,dim):
        """
        Inizialize a vector x to be compatible with the range/domain of M.
        If dim == "noise" initialize x to be compatible with the size of white noise used for sampling.
        """
        if dim == "noise":
            self.prior.sqrtM.init_vector(x,1)
        else:
            self.prior.init_vector(x,dim)
    
    def generate_vector(self,dim=0,v=None):
        """
        Generate/initialize a dolfin generic vector to be compatible with the size of dof.
        """
        vec = dl.Vector()
        self.init_vector(vec,dim)
        if v is None:
            vec.zero()
        else:
            vec[:]=v
        return vec
    
    def cost(self,x):
        """
        The whitened prior potential
        """
        Mx = self.generate_vector()
        self.prior.M.mult(x,Mx)
        return .5*Mx.inner(x)
    
    def grad(self,x,out=None):
        """
        The gradient of whitened prior potential
        """
        if out is None:
            out=self.generate_vector()
            rtrn=True
        else:
            rtrn=False 
        out.zero()
        self.prior.M.mult(x,out)
        if rtrn: return out
    
    def sample(self,noise=None,out=None):
        """
        Sample a random function v(x) ~ N(0,I)
        vector v ~ N(0,M^(-1))
        """
        if noise is None:
            noise = self.generate_vector(dim="noise")
            parRandom.normal(1., noise)
        rhs=self.prior.sqrtM*noise
        if out is None:
            out = self.generate_vector(dim=0)
            rtrn=True
        else:
            rtrn=False 
        self.prior.Msolver.solve(out,rhs)
        if rtrn: return out
    
    def C_act(self,u_actedon,Cu=None,comp=1,transp=False):
        """
        Calculate operation of C^comp on vector a: a --> C^comp * a
        """
        if Cu is None:
            Cu=self.generate_vector()
            rtrn=True
        else:
            rtrn=False
        if comp==0:
            Cu[:]=u_actedon
        else:
            if comp in [0.5,1]:
                solver=getattr(self.prior,{0.5:'A',1:'R'}[comp]+'solver')
                if not transp:
                    Mu=self.generate_vector()
                    self.prior.M.mult(u_actedon,Mu)
                    solver.solve(Cu,Mu)
                else:
                    inv_u=self.generate_vector(dim=1)
                    solver.solve(inv_u,u_actedon)
                    self.prior.M.mult(inv_u,Cu)
            elif comp in [-0.5,-1]:
                multer=getattr(self.prior,{-0.5:'A',-1:'R'}[comp])
                if not transp:
                    _u=self.generate_vector()
                    multer.mult(u_actedon,_u)
                    self.prior.Msolver.solve(Cu,_u)
                else:
                    invMu=self.generate_vector(dim=1)
                    self.prior.Msolver.solve(invMu,u_actedon)
                    multer.mult(invMu,Cu)
            else:
                warnings.warn('Action not defined!')
                Cu=None; pass
        if rtrn: return Cu
    
    def u2v(self,u,dord=0,v=None,u_ref=None,q=None):
        """
        Transform the original parameter u to the whitened parameter v:=Lmd^(-1)(u-u_ref)
        """
        if u_ref is None: u_ref=self.prior.mean
        if q is None: q = self.prior.q
        u_=u.copy()
        u_.axpy(-1.,u_ref)
        if v is None:
            v=self.generate_vector(dim=0)
            rtrn=True
        else:
            rtrn=False
        self.C_act(u_,v,comp=-0.5)
        nm_v=v.norm('l2')
        if dord==0:
            v*=nm_v**(q/2-1)
            if rtrn: return v
        if dord==1:
            def grad(w, Jw=None, adj=False):
                u_.zero()
                if Jw is None:
                    Jw=self.generate_vector(dim=1)
                    rtrn=True
                else:
                    rtrn=False
                if adj:
                    u_.axpy(v.inner(w)*nm_v**(q/2-3)*(q/2-1),v)
                    u_.axpy(nm_v**(q/2-1),w)
                    self.C_act(u_,Jw,comp=-0.5,transp=True)
                else:
                    self.C_act(w,u_,comp=-0.5)
                    Jw.axpy(v.inner(u_)*nm_v**(q/2-3)*(q/2-1),v)
                    Jw.axpy(nm_v**(q/2-1),u_)
                if rtrn: return Jw
            return grad
    
    def v2u(self,v,dord=0,u=None,u_ref=None,q=None):
        """
        Transform the whitened parameter v back to the original parameter u=Lmd(v)+u_ref
        """
        if u_ref is None: u_ref=self.prior.mean
        if q is None: q = self.prior.q
        nm_v=v.norm('l2')
        if u is None:
            u=self.generate_vector(dim=1)
            rtrn=True
        else:
            rtrn=False
        if dord==0:
            self.C_act(v*nm_v**(2/q-1),u,comp=0.5)
            u.axpy(1.,u_ref)
            if rtrn: return u
        if dord==1:
            def grad(w, Jw=None, adj=False):
                if Jw is None:
                    Jw=self.generate_vector(dim=1);
                    rtrn=True
                else:
                    rtrn=False
                hlp=Jw.copy()
                if adj:
                    self.C_act(w,hlp,comp=0.5,transp=True)
                    Jw.axpy(v.inner(hlp)*nm_v**(2/q-3)*(2/q-1),v)
                    Jw.axpy(nm_v**(2/q-1),hlp)
                else:
                    hlp.axpy(v.inner(w)*nm_v**(2/q-3)*(2/q-1),v)
                    hlp.axpy(nm_v**(2/q-1),w)
                    self.C_act(hlp,Jw,comp=0.5)
                if rtrn: return Jw
            return grad
        if dord==2:
            def hess(w,m,mHw=None,adj=False):
                if mHw is None:
                    mHw=self.generate_vector(dim=1); 
                    rtrn=True
                else:
                    rtrn=False
                hlp1=mHw.copy(); hlp2=mHw.copy()
                self.C_act((2/q-1)*m*v.inner(w)*nm_v**(2/q-3),mHw,comp=0.5,transp=adj)
                self.C_act((2/q-1)*v*nm_v**(2/q-3),hlp1,comp=0.5)
                hlp2.axpy(v.inner(w)*nm_v**(2/q-5)*(2/q-3),v)
                hlp2.axpy(nm_v**(2/q-3),w)
                self.C_act((2/q-1)*hlp2.copy(),hlp2,comp=0.5)
                if adj:
                    mHw.axpy(m.inner(hlp1),w)
                    mHw.axpy(m.inner(hlp2),v)
                else:
                    mHw.axpy(m.inner(w),hlp1)
                    mHw.axpy(m.inner(v),hlp2)
                if rtrn: return mHw
            return hess
    
    def jacdet(self,v,dord=0,q=None):
        """
        (log) Jacobian determinant log dLmd
        """
        if q is None: q = self.prior.q
        nm_v=v.norm('l2')
        if dord==0:
            return (2/q-1)*self.prior.dim*np.log(nm_v)
        if dord==1:
            return (2/q-1)*self.prior.dim*(v/nm_v**2)

class wht_Hessian:
    """
    Whiten the given Hessian H(u) to be H(v) := dv2u' * H(u) * dv2u
    """
    def __init__(self,parameter,whtprior,HessApply,**kwargs):
        self.parameter=parameter
        self.whtprior=whtprior
        self.HessApply=HessApply
        self.post_grad=kwargs.pop('post_grad',whtprior.prior.grad)
    
    def mult(self,x,y):
        Jx = self.whtprior.v2u(self.parameter,1)(x)
        hlp = self.whtprior.generate_vector()
        self.HessApply.mult(Jx,hlp)
        self.whtprior.v2u(self.parameter,1)(hlp, Jw=y,adj=True) # GNH approximation
        # # g = self.whtprior.prior.gen_vector()
        # # self.post_grad(self.whtprior.v2u(self.parameter), g) # should be misfit.grad instead
        # g = self.post_grad(self.whtprior.v2u(self.parameter))
        # y.axpy(1, self.whtprior.v2u(self.parameter, 2)(x, g,adj=True) ) # exact Hessian with it
    
    def inner(self,x,y):
        Hy = self.whtprior.generate_vector()
        Hy.zero()
        self.mult(y,Hy)
        return x.inner(Hy)

if __name__ == '__main__':
    from pde import *
    from prior import _prior
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
    # prior.mean=prior.gen_vector()
    whtprior = wht_prior(prior)
    
    # test
    h=1e-6; z=whtprior.sample(); v=whtprior.sample(); w=whtprior.sample();
    print('**** Testing v2u (Lmd) ****')
    val,grad,hess=whtprior.v2u(z,0),whtprior.v2u(z,1),whtprior.v2u(z,2)
    val1,grad1=whtprior.v2u(z+h*v,0),whtprior.v2u(z+h*w,1)
    gradv_fd=(val1-val)/h
    gradv=grad(v)
    rdiff_gradv=(gradv_fd-gradv).norm('l2')/v.norm('l2')
    print('Relative error in gradient: %0.8f' %(rdiff_gradv))
    hessv_fd=(grad1(v)-gradv)/h
    hessv=hess(v,w)
    rdiff_hessv=(hessv_fd-hessv).norm('l2')/(v.norm('l2')*w.norm('l2'))
    print('Relative error in Hessian: %0.8f' %(rdiff_hessv))
    
    h=1e-6; u=prior.sample(); v=prior.sample()
    print('\n**** Testing u2v (invLmd) ****')
    val,grad=whtprior.u2v(u,0),whtprior.u2v(u,1)
    val1=whtprior.u2v(u+h*v,0)
    gradv_fd=(val1-val)/h
    gradv=grad(v)
    rdiff_gradv=(gradv_fd-gradv).norm('l2')/v.norm('l2')
    print('Relative error in gradient: %0.8f' %(rdiff_gradv))
    u1=whtprior.v2u(val,0)
    rdiff_Lmd_iLmd=(u1-u).norm('l2')/u.norm('l2')
    print('Relative error of Lmd-invLmd in a random direction between composition and identity: %.10f' % (rdiff_Lmd_iLmd) )
    u2=whtprior.u2v(whtprior.v2u(u,0),0)
    rdiff_iLmd_Lmd=(u2-u).norm('l2')/u.norm('l2')
    print('Relative error of invLmd-Lmd in a random direction between composition and identity: %.10f' % (rdiff_iLmd_Lmd))
    v1=whtprior.v2u(val,1)(gradv)
    rdiff_dLmd_diLmd=(v1-v).norm('l1')/v.norm('l2')
    print('Relative error of dLmd-dinvLmd in a random direction between composition and identity: %.10f' % (rdiff_dLmd_diLmd))
    v2=whtprior.u2v(u,1)(whtprior.v2u(val,1)(v))
    rdiff_diLmd_dLmd=(v2-v).norm('l1')/v.norm('l2')
    print('Relative error of dinvLmd-dLmd in a random direction between composition and identity: %.10f' % (rdiff_diLmd_dLmd))
    
    h=1e-6; z=whtprior.sample(); v=whtprior.sample();
    print('\n**** Testing jacdet ****')
    val,grad=whtprior.jacdet(z,0),whtprior.jacdet(z,1)
    val1=whtprior.jacdet(z+h*v,0)
    djacdet_fd=(val1-val)/h
    djacdet=grad.inner(v)
    rdiff_jacdet=np.abs(djacdet_fd-djacdet)/v.norm('l2')
    print('error in gradient of Jacobian determinant: %0.8f' %(rdiff_jacdet))
    
    # # check conversions
    # print('\n**** Checking conversions ****')
    # u = prior.sample()
    # cost_u=prior.cost(u)
    # grad_u=prior.gen_vector()
    # prior.grad(u,grad_u)
    # print('The prior potential at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(cost_u,grad_u.norm('l2')))
    # fig=dl.plot(vector2Function(u,prior.Vh))
    # plt.colorbar(fig); plt.show()
    # v = whtprior.u2v(u)
    # cost_v=whtprior.cost(v)
    # grad_v=prior.gen_vector()
    # whtprior.grad(v,grad_v)
    # print('The potential of whitened prior at v is %0.4f, and the L2 norm of its gradient is %0.4f' %(cost_v,grad_v.norm('l2')))
    # fig=dl.plot(vector2Function(v,prior.Vh))
    # plt.colorbar(fig); plt.show()
    #
    # v = whtprior.sample()
    # cost_v=whtprior.cost(v)
    # whtprior.grad(v,grad_v)
    # print('The potential of whitened prior at v is %0.4f, and the L2 norm of its gradient is %0.4f' %(cost_v,grad_v.norm('l2')))
    # fig=dl.plot(vector2Function(v,prior.Vh))
    # plt.colorbar(fig); plt.show()
    # u = whtprior.v2u(v)
    # cost_u=prior.cost(u)
    # prior.grad(u,grad_u)
    # print('The prior potential at u is %0.4f, and the L2 norm of its gradient is %0.4f' %(cost_u,grad_u.norm('l2')))
    # fig=dl.plot(vector2Function(u,prior.Vh))
    # plt.colorbar(fig); plt.show()