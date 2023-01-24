#!/usr/bin/env python
"""
Kernel defined by series expansion
-- with basis choices 'wavelet' and 'Fourier'
-- classical operations in high dimensions
---------------------------------------------------------------
Shiwei Lan @ ASU, 2022
----------------------
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@gmail.com;"

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
# self defined modules
from .linalg import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Ker:
    def __init__(self,x,L=None,store_eig=False,**kwargs):
        """
        Initialize the kernel class with inputs and settings
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
        self.parameters=kwargs # all parameters of the kernel
        self.basis_opt=self.parameters.get('basis_opt','Fourier') # basis option
        if self.basis_opt=='wavelet':
            self.wvlet_typ=self.parameters.get('wvlet_typ','Harr') # wavelet type
        self.sigma2=self.parameters.get('sigma2',1) # magnitude
        self.l=self.parameters.get('l',0.5) # correlation length
        self.s=self.parameters.get('s',2) # smoothness
        self.q=self.parameters.get('q',1) # norm power
        self.jit=self.parameters.get('jit',1e-6) # jitter
        self.N,self.d=self.x.shape # size and dimension
        if L is None:
            L=min(self.N,100)
        self.L=L # truncation in Karhunen-Loeve expansion
        if self.L>self.N:
            warnings.warn("Karhunen-Loeve truncation number cannot exceed size of the discrete basis!")
            self.L=self.N
        try:
            self.comm=MPI.COMM_WORLD
        except:
            print('Parallel environment not found. It may run slowly in serial.')
            self.comm=None
        self.spdapx=self.parameters.get('spdapx',self.N>1e3)
        self.store_eig=store_eig
        if self.store_eig:
            # obtain partial eigen-basis
            self.eigv,self.eigf=self.eigs(**kwargs)
    
    def _Fourier(self, x=None, L=None, **kwargs):
        """
        Fourier basis
        """
        if x is None:
            x=self.x
        if L is None:
            L=self.L
        if self.d==1:
            f=np.cos(np.pi*x*np.arange(L)); f[:,0]/=np.sqrt(2) # (N,L)
        elif self.d==2:
            rtL=int(np.sqrt(L))
            f=np.cos(np.pi*x[:,[0]]*(np.arange(rtL)+0.5))[:,:,None]*np.cos(np.pi*x[:,[1]]*(np.arange(rtL)+0.5))[:,None,:] # (N,rtL,rtL)
            f=f.reshape((-1,rtL**2))
            resL=L-rtL**2
            if resL>0:
                f=np.append(f,np.cos(np.pi*x[:,[0]]*(rtL+0.5))*np.cos(np.pi*x[:,[1]]*(np.arange(min(rtL,resL))+0.5)),axis=1) # row convention (type='C')
                if resL>rtL:
                    f=np.append(f,np.cos(np.pi*x[:,[0]]*(np.arange(resL-rtL)+0.5))*np.cos(np.pi*x[:,[1]]*(rtL+0.5)),axis=1)
            f*=2/np.sqrt(self.N) # (N,L)
        else:
            raise NotImplementedError('Basis for spatial dimension larger than 2 is not implemented yet!')
        return f
    
    def _Mexican_hat(self, x=None, L=None, sigma=1):
        """
        Mexican hat wavelet, a.k.a. the Ricker wavelet
        """
        if x is None:
            x=self.x
        if L is None:
            L=self.L
        if self.d==1:
            psi=lambda x: 2/(np.sqrt(3*sigma)*np.pi**(1/4))*(1-(x/sigma)**2)*np.exp(-x**2/(2*sigma**2))
            psi_jk=lambda x, j, k=0: 2**(j/2) * psi(2**j * x - k)
        elif self.d==2:
            psi2=lambda x: 1/(np.pi*sigma**4)*(1-1/2*(np.sum(x**2,axis=1)/sigma**2))*np.exp(-np.sum(x**2,axis=1)/(2*sigma**2))
            psi_jk=lambda x, j, k=0: 2**(j/2) * psi2(2**j * x - k)
        else:
            raise NotImplementedError('Basis for spatial dimension larger than 2 is not implemented yet!')
        n=int(np.log2(L))
        f=np.empty((self.N,0))
        for j in range(n):
            f=np.append(f,np.stack([psi_jk(x, j, k) for k in range(2**j)]).T,axis=-1)
        if L>2**n:
            f=np.append(f,np.stack([psi_jk(x, n, k) for k in range(L+1-2**n)]).T,axis=-1) # (N,L)
        f/=np.linalg.norm(f,axis=0)
        return f
    
    def _wavelet(self, x=None, L=None, d=None, **kwargs):
        """
        Common wavelet bases including Harr, Shannon, Meyer, Mexican Hat, Poisson, etc.
        """
        if x is None:
            x=self.x
        if L is None:
            L=self.L
        if d is None:
            d=self.d
        if self.wvlet_typ=='Harr':
            phi=lambda x: 1.0*(x>=0)*(x<1)
            psi=lambda x: 1.0*(x>=0)*(x<0.5) - 1.0*(x>=0.5)*(x<1) # Harr wavelet
        elif self.wvlet_typ=='Shannon':
            phi=np.sinc
            psi=lambda x: 2*np.sinc(2*x)-np.sinc(x) # Shannon wavelet
        elif self.wvlet_typ=='Meyer':
            phi=lambda x: ( 2/3*np.sinc(2/3*x) + 4/(3*np.pi)*np.cos(4*np.pi/3*x) )/( 1 - 16/9*x**2 )
            psi1=lambda x: ( 4/(3*np.pi)*np.cos(2*np.pi/3*(x-1/2)) - 4/3*np.sinc(4/3*(x-1/2)) )/( 1-16/9*(x-1/2)**2 )
            psi2=lambda x: ( 8/(3*np.pi)*np.cos(8*np.pi/3*(x-1/2)) + 4/3*np.sinc(4/3*(x-1/2)) )/( 1-64/9*(x-1/2)**2 )
            psi=lambda x: psi1(x)+psi2(x)
        elif self.wvlet_typ=='MexHat':
            # return self._Mexican_hat(x, L, kwargs.pop('sigma',1))
            phi=lambda x: np.ones((self.N,1))
            sigma=kwargs.pop('sigma',1)
            psi=lambda x: 2/(np.sqrt(3*sigma)*np.pi**(1/4))*(1-(x/sigma)**2)*np.exp(-x**2/(2*sigma**2))
        elif self.wvlet_typ=='Poisson':
            phi=lambda x: np.ones((self.N,1))
            psi=lambda x: 1/np.pi*(1-x**2)/(1+x**2)**2
        else:
            raise ValueError('Wavelet type not implemented!')
        psi_jk=lambda x, j, k=0: 2**(j/2) * psi(2**j * x - k)
        # if d==2: psi_jk=lambda x, j, k=0: 2**(j) * psi(2**j * x[:,[0]] - k) * psi(2**j * x[:,[1]] - k)
        if d==1:
            n=int(np.log2(L))
            f=phi(x)
            for j in range(n):
                f=np.append(f,psi_jk(x, j, np.arange(2**j)),axis=-1)
            if L>2**n:
                f=np.append(f,psi_jk(x, n, np.arange(L-2**n)),axis=-1) # (N,L)
        elif d==2:
            rtL=int(np.sqrt(L))
            # rtL=np.ceil(np.sqrt(L)).astype('int')
            f=self._wavelet(x[:,[0]], rtL, d=1)[:,:,None]*self._wavelet(x[:,[1]], rtL, d=1)[:,None,:] # (N,rtL,rtL)
            f=f.reshape((-1,rtL**2))
            # f=f[:,:L]
            resL=L-rtL**2
            if resL>0:
                # f=np.append(f,psi_jk(x[:,[0]],j=int(np.ceil(np.log2(rtL))),k=np.arange(min(rtL,resL)))*self._wavelet(x[:,[1]],min(rtL,resL),d=1),axis=1) # row convention (type='C')
                f=np.append(f,psi_jk(x[:,[0]],j=int(np.ceil(np.log2(rtL))),k=rtL-2**int(np.ceil(np.log2(rtL))))*self._wavelet(x[:,[1]],min(rtL,resL),d=1),axis=1)
                if resL>rtL:
                    # f=np.append(f,self._wavelet(x[:,[0]],resL-rtL,d=1)*psi_jk(x[:,[1]],j=int(np.ceil(np.log2(rtL))),k=np.arange(resL-rtL)),axis=1)
                    f=np.append(f,self._wavelet(x[:,[0]],resL-rtL,d=1)*psi_jk(x[:,[1]],j=int(np.ceil(np.log2(rtL))),k=rtL-2**int(np.ceil(np.log2(rtL)))),axis=1)
            # f/=np.linalg.norm(f,axis=0)
            f/=np.sqrt(self.N)
        else:
            raise NotImplementedError('Basis for spatial dimension larger than 2 is not implemented yet!')
        # f/=np.linalg.norm(f,axis=0)
        return f
    
    def _qrteigv(self, L=None):
        """
        Decaying (q-root) eigenvalues
        """
        if L is None:
            L=self.L
        if self.d==1:
            gamma=(self.l+np.arange(L))**(-(self.s/self.d+1./2-1./self.q))
        elif self.d==2:
            rtL=int(np.sqrt(L))
            gamma=(self.l+(np.arange(rtL)+0.5)[:,None]**2+(np.arange(rtL)+0.5)**2)**(-(self.s/self.d+1./2-1./self.q))
            gamma=gamma.flatten()
            resL=L-rtL**2
            if resL>0:
                gamma=np.append(gamma,(self.l+(rtL+0.5)**2+(np.arange(min(rtL,resL))+0.5)**2)**(-(self.s/self.d+1./2-1./self.q)))
                if resL>rtL:
                    gamma=np.append(gamma,(self.l+(np.arange(resL-rtL)+0.5)**2+(rtL+0.5)**2)**(-(self.s/self.d+1./2-1./self.q)))
        else:
            raise NotImplementedError('Basis for spatial dimension larger than 2 is not implemented yet!')
        gamma*=self.sigma2**(1/self.q)
        return gamma
    
    def tomat(self,**kwargs):
        """
        Get kernel as matrix
        """
        alpha=kwargs.get('alpha',1)
        eigv,eigf = self.eigs() # obtain eigen-basis
        if alpha<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
        C = (eigf*pow(eigv,alpha)).dot(eigf.T) + self.jit*sps.eye(self.N)
        if type(C) is np.matrix:
            C=C.getA()
        if self.spdapx and not sps.issparse(C):
            warnings.warn('Possible memory overflow!')
        return C
    
    def mult(self,v,**kwargs):
        """
        Kernel multiply a function (vector): C*v
        """
        alpha=kwargs.pop('alpha',1) # power of dynamic eigenvalues
        transp=kwargs.get('transp',False) # whether to transpose the result
        if not self.spdapx:
            Cv=multf(self.tomat(alpha=alpha),v,transp)
        else:
            eigv,eigf = self.eigs() # obtain eigen-pairs
            if alpha<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
            eigv_ = pow(eigv,alpha)
            prun=kwargs.get('prun',True) and self.comm # control of parallel run
            if prun:
                try:
#                     import pydevd; pydevd.settrace()
                    nproc=self.comm.size; rank=self.comm.rank
                    if nproc==1: raise Exception('Only one process found!')
                    Cv_loc=multf(eigf[rank::nproc,:]*eigv_,multf(eigf.T,v,transp),transp)
                    Cv=np.empty_like(v)
                    self.comm.Allgatherv([Cv_loc,MPI.DOUBLE],[Cv,MPI.DOUBLE])
                    pidx=np.concatenate([np.arange(self.N)[i::nproc] for i in np.arange(nproc)])
                    Cv[pidx]=Cv.copy()
                except Exception as e:
                    if rank==0:
                        warnings.warn('Parallel run failed: '+str(e))
                    prun=False
                    pass
            if not prun:
            #     Cv=np.concatenate([multf(eigf_i*eigv_,multf(eigf.T,v,transp),transp) for eigf_i in eigf])
            # # if transp: Cv=Cv.swapaxes(0,1)
                Cv=multf(eigf*eigv_,multf(eigf.T,v,transp),transp)
            Cv+=self.jit*v
        return Cv
    
    def solve(self,v,**kwargs):
        """
        Kernel solve a function (vector): C^(-1)*v
        """
        transp=kwargs.pop('transp',False)
        if not self.spdapx:
            invCv=mdivf(self.tomat(),v,transp)
        else:
            invCv=self.mult(v,alpha=-1,transp=transp)
        return invCv
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis
        C * eigf_i = eigf_i * eigv_i, i=1,...,L
        """
        if L is None:
            L=self.L;
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            basisf=getattr(self,'_'+self.basis_opt) # obtain basis function
            eigf=basisf(x=self.x, L=L, **kwargs)
            if eigf.shape[1]<L:
                eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
                warnings.warn('zero eigenvectors padded!')
            eigv=abs(self._qrteigv(L))**self.q
            if len(eigv)<L:
                eigv=np.pad(eigv,[0,L-len(eigv)],mode='constant')
                warnings.warn('zero eigenvalues padded!')
        else:
            eigv,eigf=self.eigv,self.eigf
            eigv=eigv[:L]; eigf=eigf[:,:L]
        return eigv,eigf
    
    def act(self,x,alpha=1,**kwargs):
        """
        Obtain the action of C^alpha
        y=C^alpha *x
        """
        transp=kwargs.get('transp',False)
        if alpha==0:
            y=x
        else:
            y=self.mult(x,alpha=alpha,**kwargs)
        return y
    
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

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    meshsz=32
    # x=np.stack([np.sort(np.random.rand(meshsz**2)),np.sort(np.random.rand(meshsz**2))]).T
    xx,yy=np.meshgrid(np.linspace(0,1,meshsz),np.linspace(0,1,meshsz))
    x=np.stack([xx.flatten(),yy.flatten()]).T
    ker=Ker(x,L=1000,store_eig=True,basis_opt='wavelet', l=1, q=1.0) # constrast with q=2.0
    verbose=ker.comm.rank==0 if ker.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(ker.eigv[:min(10,ker.L)],4))
        print('Eigenvectors :', np.round(ker.eigf[:,:min(10,ker.L)],4))

    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))

    v=np.random.rand(meshsz**2,2)
    C=ker.tomat()
    Cv=C.dot(v)
    Cv_te=ker.act(v)
    if verbose:
        print('Relative difference between matrix-vector product and action on vector: {:.4f}'.format(spla.norm(Cv-Cv_te)/spla.norm(Cv)))

    t2=time.time()
    if verbose:
        print('time: %.5f'% (t2-t1))

    solver=spsla.spsolve if sps.issparse(C) else spla.solve
    invCv=solver(C,v)
#     C_op=spsla.LinearOperator((ker.N,)*2,matvec=lambda v:ker.mult(v))
#     invCv=spsla.cgs(C_op,v)[0][:,np.newaxis]
    invCv_te=ker.act(v,-1)
    if verbose:
        print('Relatively difference between direct solver and iterative solver: {:.4f}'.format(spla.norm(invCv-invCv_te)/spla.norm(invCv)))
    
    t3=time.time()
    if verbose:
        print('time: %.5f'% (t3-t2))