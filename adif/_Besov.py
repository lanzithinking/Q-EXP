'''
Besov prior of the Advection-Diffusion problem written in FEniCS-2019.1.0 and hIPPYlib-3.0
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
__version__ = "0.3"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import dolfin as dl
import ufl
import numpy as np
from scipy.stats import gennorm
import matplotlib.pyplot as plt

import os,sys
sys.path.append( "../" )
from util.kernel.serexp import Ker
# sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *
from hippylib.modeling.prior import SqrtPrecisionPDE_Prior
# from randomizedEigensolver_ext import *

import warnings
warnings.simplefilter('once')

class _Besov(SqrtPrecisionPDE_Prior,Ker):
    """
    Besov prior B(mu,C), assembled(C^(-1)) = R = A * M^(-1) * A
    C = A^{-1} M A^-1
    """
    def __init__(self, Vh, gamma = 1., delta = 8., **kwargs):
        """
        Initialize the BiLaplacian prior
        """
        # function space
        self.Vh = Vh
        self.dim=Vh.dim()
        self.mpi_comm = self.Vh.mesh().mpi_comm()
        self.rank = dl.MPI.rank(self.mpi_comm)
        # prior parameters
        self.gamma = gamma
        self.delta = delta
        assert delta != 0., "Intrinsic Gaussian Prior are not supported"
        Theta = kwargs.pop('Theta',None)
        robin_bc = kwargs.pop('robin_bc',True)
        def sqrt_precision_varf_handler(trial, test): 
            if Theta == None:
                varfL = ufl.inner(ufl.grad(trial), ufl.grad(test))*ufl.dx
            else:
                varfL = ufl.inner( Theta*ufl.grad(trial), ufl.grad(test))*ufl.dx
            varfM = ufl.inner(trial,test)*ufl.dx
            varf_robin = ufl.inner(trial,test)*ufl.ds
            if robin_bc:
                robin_coeff = gamma*np.sqrt(delta/gamma)/1.42
            else:
                robin_coeff = 0.
            return dl.Constant(gamma)*varfL + dl.Constant(delta)*varfM + dl.Constant(robin_coeff)*varf_robin
        # define mean
        mean = kwargs.pop('mean', dl.interpolate(dl.Constant(0.25), self.Vh).vector())
        SqrtPrecisionPDE_Prior.__init__(self,self.Vh, sqrt_precision_varf_handler, mean)
        # if self.rank == 0:
        #     print( "Prior regularization: (delta - gamma*Laplacian)^order: delta={0}, gamma={1}, order={2}".format(self.delta, self.gamma,2) )
        self.q=kwargs.get('q',1)
        self.L=kwargs.pop('L',100)
        self.store_eig=kwargs.pop('store_eig',False)
        Ker.__init__(self,x=self.Vh.tabulate_dof_coordinates(),L=self.L,store_eig=self.store_eig,sigma2=1./self.delta,l=self.gamma/self.delta,**kwargs)
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self._eigs(upd=True,**kwargs)
    
    # def init_vector(self,x,dim):
    #     """
    #     Inizialize a vector :code:`x` to be compatible with the range/domain of :math:`R`.
    #
    #     If :code:`dim == "noise"` inizialize :code:`x` to be compatible with the size of
    #     white noise used for sampling.
    #     """
    #     if dim == "noise":
    #         self.sqrtM.init_vector(x, 1)
    #     else:
    #         self.M.init_vector(x,dim)
    
    def gen_vector(self, v=None):
        """
        generic function to generate a vector in the PARAMETER space
        """
        if v is None:
            vec = dl.Vector(self.mpi_comm)
            self.init_vector(vec, 0)
        else:
            if isinstance(v, dl.Vector):
                vec = dl.Vector(v)
            elif isinstance(v, np.ndarray):
                vec = self.gen_vector()
                vec.set_local(v)
            else:
                df.warning('Unknown type.')
                vec=None
        return vec
    
    # def _eigs(self, L=None, upd=False, **kwargs):
    #     """
    #     get partial eigen-pairs of the covariance kernel
    #     """
    #     if L is None:
    #         L=self.L
    #     if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
    #         k=kwargs['k'] if 'k' in kwargs else kwargs['incr_k'] if 'incr_k' in kwargs else L
    #         p=kwargs['p'] if 'p' in kwargs else 20
    #         if len(kwargs)==0:
    #             kwargs['k']=k
    #         Omega = MultiVector(self.gen_vector(), k+p)
    #         parRandom.normal(1., Omega)
    #         # eigv,eigf = singlePassGx(self.R, self.M, self.Msolver, Omega, **kwargs)
    #         eigv,eigf = singlePass(self.R, Omega, k)
    #     else:
    #         eigv,eigf=self.eigv,self.eigf
    #         if L<len(eigv):
    #             eigv=eigv[:L]; eigf_=MultiVector(self.gen_vector(),L)
    #             for j in range(L):
    #                 eigf_[j]=eigf[j]
    #             eigf=eigf_
    #     return eigv,eigf
    
    def _eigs(self, L=None, upd=False, **kwargs):
        """
        get partial eigen-pairs of the covariance kernel
        """
        if L is None:
            L=self.L
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            # k=kwargs['k'] if 'k' in kwargs else kwargs['incr_k'] if 'incr_k' in kwargs else L
            # p=kwargs['p'] if 'p' in kwargs else 20
            # if len(kwargs)==0:
            #     kwargs['k']=k
            # Omega = MultiVector(self.gen_vector(), k+p)
            # parRandom.normal(1., Omega)
            # # eigv,eigf = singlePassGx(self.R, self.M, self.Msolver, Omega, **kwargs)
            # eigv,eigf = singlePass(self.R, Omega, k)
            eigv,eigf_ = Ker.eigs(self,L=L,upd=upd,**kwargs)
            eigf = MultiVector(self.gen_vector(), eigf_.shape[1])
            for j in range(eigf_.shape[1]): eigf[j].set_local(eigf_[:,j])
        else:
            eigv,eigf=self.eigv,self.eigf
            if L<len(eigv):
                eigv=eigv[:L]; eigf_=MultiVector(self.gen_vector(),L)
                for j in range(L):
                    eigf_[j]=eigf[j]
                eigf=eigf_
        return eigv,eigf
    
    def cost(self, u):
        """
        negative log-prior
        """
        du = u - self.mean
        eigv,eigf=self._eigs()
        proj=eigf.dot(du)
        reg = .5*np.sum(abs(proj)**self.q/eigv)
        return reg
    
    def grad(self, u, out):
        """
        gradient of negative log-prior
        """
        out.zero()
        du = u - self.mean
        eigv,eigf=self._eigs()
        proj=eigf.dot(du)
        eigf.reduce(out, 0.5*self.q*abs(proj)**(self.q-1) *np.sign(proj)/eigv)
    
    def applyR(self, dm, out):
        """
        apply C^{-1}: out = C{-1} dm
        """
        eigv,eigf=self._eigs()
        proj=eigf.dot(dm)
        eigf.reduce(out,proj/eigv)
    
    def hess(self, u, v, out):
        """
        Hessian of negative log-prior
        """
        out.zero()
        du = u - self.mean
        eigv,eigf=self._eigs()
        proj=eigf.dot(du)
        eigf.reduce(out, 0.5*self.q*(self.q-1)* (abs(proj)**(self.q-2)/eigv)*eigf.dot(v))
    
    def sample(self, whiten=False, add_mean=False):
        """
        Generate a prior sample
        """
        epd_rv=gennorm.rvs(beta=self.q,scale=2**(1.0/self.q),size=self.L) # (L,)
        eigv,eigf=self._eigs()
        u_vec = dl.Vector()
        self.init_vector(u_vec, 0)
        eigf.reduce(u_vec, eigv**(1/self.q*(not whiten))*epd_rv)
        
        if add_mean:
            u_vec.axpy(1., self.mean)
        return u_vec
    
    def logpdf(self, u, whiten=False, add_mean=False, grad=False):
        """
        Compute the logarithm of prior density (and its gradient) of function u.
        """
        u_m=dl.Vector(u)
        if add_mean:
            u_m.axpy(-1.,self.u2v(self.mean) if whiten else self.mean)
        
        eigv,eigf=self._eigs()
        proj=eigf.dot(u_m)
        
        logpri=-0.5*np.sum(abs(proj)**self.q/eigv**(not whiten)) #+np.sum(np.log(abs(eigv)))/self.q
        if grad:
            gradpri=dl.Vector()
            self.init_vector(gradpri,0)
            eigf.reduce(gradpri, -0.5*self.q*abs(proj)**(self.q-1) *np.sign(proj)/eigv**(not whiten))
            return logpri,gradpri
        else:
            return logpri
    
    def C_act(self, u_actedon, comp=1, transp=False):
        """
        Compute operation of C^comp on vector u: u --> C^comp * u
        """
        if isinstance(u_actedon, np.ndarray):
            u_actedon=self.gen_vector(u_actedon)
        if comp==0:
            return u_actedon
        else:
            Cu=dl.Vector()
            self.init_vector(Cu, np.ceil((np.sign(comp)+1)/2).astype('int'))
            if comp==-1:
                self.applyR(u_actedon, Cu)
            else:
                eigv,eigf=self._eigs()
                proj=eigf.dot(u_actedon)*eigv**(comp)
                eigf.reduce(Cu, proj)
        return Cu
    
    def u2v(self, u, u_ref=None):
        """
        v:=C^(-1/2) (u-u_ref)
        """
        v = dl.Vector()
        self.init_vector(v,1)
        b = u if u_ref is None else u - u_ref
        v = self.C_act(b, comp=-0.5)
        return v
    
    def v2u(self, v, u_ref=None):
        """
        u = u_ref + C^(1/2) v
        """
        u = dl.Vector()
        self.init_vector(u,1)
        u = self.C_act(v, comp=0.5)
        if u_ref is not None:
            u.axpy(1., u_ref)
        return u
    
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
    gamma = 2.; delta = 10.
    prior = _Besov(Vh, gamma=gamma, delta=delta, q=1.1)
    
    # test gradient and Hessian
    u=prior.sample()
    logpri,gradpri=prior.logpdf(u, add_mean=True, grad=True)
    v=prior.sample()
    hessv=prior.gen_vector()
    prior.hess(u, v, hessv)
    h=1e-3
    logpri1,gradpri1=prior.logpdf(u+h*v, add_mean=True, grad=True)
    gradv_fd=(logpri1-logpri)/h
    gradv=gradpri.inner(v)
    rdiff_gradv=np.abs(gradv_fd-gradv)/v.norm('l2')
    print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    hessv_fd=(gradpri1-gradpri)/h
    rdiff_hessv=(hessv_fd+hessv).norm('l2')/v.norm('l2')
    print('Relative difference of Hessian-action in a random direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
    
    # tests
    whiten=False
    # u=prior.sample(whiten=whiten)
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