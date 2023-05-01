#!/usr/bin/env python
"""
Kernel defined by graph Laplacian
-- with kernel choices 'powered exponential' and 'Matern class'
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
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import img_to_graph as i2g
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
        x: inputs, path to image (string) or image (numpy array)
        L: truncation number in Mercer's series
        store_eig: indicator to store eigen-pairs, default to be false
        sigma2: magnitude, default to be 1
        tau2; (inverse) length-scale of the kernel, default to be 1
        l: correlation length of the weighting function, default to be 0.5
        s: smoothness, default to be 2
        jit: jittering term, default to be 1e-6
        spdapx: use speed-up or approximation
        """
        self.x=self._procimg(x,kwargs.pop('format',None)) if isinstance(x, str) else x # input image
        self.parameters=kwargs # all parameters of the kernel
        self.sigma2=self.parameters.get('sigma2',1) # magnitude of the kernel
        self.tau2=self.parameters.get('tau2',1) # inverse length-scale of the kernel
        self.l=self.parameters.get('l',0.5) # correlation length for weighting function
        self.s=self.parameters.get('s',2) # smoothness of the kernel
        self.jit=self.parameters.get('jit',1e-6) # jitter
        self.imsz=self.x.shape # image size
        self.g=i2g(self.x) # compressed-sparse graph, with shape (N, N)
        self.N=self.g.shape[0] # number of nodes
        self.d=self.g.ndim
        self.Lap=self._get_Laplacian(self.g, self.parameters.get('weightedge',False), self.parameters.get('normalize',False)) # graph Lapalcian
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
    
    def _procimg(self,path2img,format=None):
        """
        Process image to output a connection graph
        """
        # read image from given location
        img=plt.imread(path2img,format)
        # convert RGB image into grayscale
        if img.ndim==3 and img.shape[-1]>=3: img=np.sum(img[:,:,:3]*np.array([0.2989,0.5870,0.1140]),axis=img.ndim-1)
        return img
    
    def _get_Laplacian(self,graph=None,weightedge=False,normalize=False):
        """
        Get graph Laplacian from the connection graph
        """
        if graph is None: graph=self.g
        graph=graph/abs(graph).max()
        if weightedge:
            graph=graph.tocsr()
            graph[graph.nonzero()]=np.exp(-.5*abs(graph[graph.nonzero()].getA()/self.l)**self.s)
        Lap=sps.csgraph.laplacian(graph,normed=normalize)
        if self.d>=2: Lap/=self.N**(1-2./self.d)*np.log(self.N)**((self.d==2)/2.0+2./self.d) # adjust for the graph size
        return Lap
    
    def prec(self,alpha=1):
        """
        Get the inverse kernel as (precision) matrix P=tau2 I + Lap
        """
        P=self.tau2*sps.eye(self.N) + self.Lap
        if alpha!=1:
            if isinstance(alpha, int):
                P=P**abs(alpha)
                if alpha<0: P=spsla.inv(P)
            else:
                eigv,eigf=self.eigs(alpha=-1)
                if alpha<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                P=(eigf*pow(eigv,alpha)).dot(eigf.T)
        return P
    
    def ldet(self,**kwargs):
        """
        Get log-determinant of C=sigma2 P^-s
        """
        ldet=self.N*np.log(self.sigma2) - self.s*(sparse_ldet(self.prec()))
        return ldet
    
    def tomat(self):
        """
        Get kernel as (covariance) matrix C=sigma2 P^-s
        """
        C=self.sigma2*(self.prec(-self.s) + self.jit*sps.eye(self.N))
        if type(C) is np.matrix:
            C=C.getA()
        if self.spdapx and not sps.issparse(C):
            warnings.warn('Possible memory overflow!')
        return C
    
    def mult(self,v,**kwargs):
        """
        Kernel multiply a function (vector): C*v
        """
        transp=kwargs.get('transp',False) # whether to transpose the result
        if not self.spdapx:
            Cv=multf(self.tomat(),v,transp)
        else:
            Ps=self.prec(self.s)
            Cv=mdivf(Ps,v,transp)*self.sigma2
            Cv+=self.sigma2*self.jit*v
        return Cv
    
    def solve(self,v,**kwargs):
        """
        Kernel solve a function (vector): C^(-1)*v
        """
        transp=kwargs.pop('transp',False)
        if not self.spdapx:
            invCv=mdivf(self.tomat(),v,transp)
        else:
            Ps=self.prec(alpha=self.s)
            invCv=multf(Ps,v,transp)/self.sigma2
        return invCv
    
    def eigs(self,L=None,upd=False,**kwargs):
        """
        Obtain partial eigen-basis
        C * eigf_i = eigf_i * eigv_i, i=1,...,L
        """
        alpha=kwargs.get('alpha',1)
        if L is None:
            L=self.L;
        if upd or L>self.L or not all([hasattr(self,attr) for attr in ('eigv','eigf')]):
            maxiter=kwargs.pop('maxiter',100)
            tol=kwargs.pop('tol',self.jit)
            P_op=self.prec()# if not self.spdapx else spsla.LinearOperator((self.N,)*2,matvec=lambda v:multf(self.prec(),v))
            try:
                eigv,eigf=spsla.eigsh(P_op,min(L,P_op.shape[0]-1),which='SM',maxiter=maxiter,tol=tol)
            except Exception as divg:
                print(*divg.args)
                eigv,eigf=divg.eigenvalues,divg.eigenvectors
            if alpha!=-1:
                if alpha>0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                eigv=self.sigma2**(alpha)*pow(eigv,-self.s*alpha)
            # eigv=abs(eigv[::-1]); eigf=eigf[:,::-1]
            eigv=np.pad(eigv,(0,L-len(eigv)),mode='constant'); eigf=np.pad(eigf,[(0,0),(0,L-eigf.shape[1])],mode='constant')
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
        elif alpha==1:
            y=self.mult(x,**kwargs)
        elif alpha==-1:
            y=self.solve(x,**kwargs)
        else:
            chol= (abs(alpha)==0.5) and kwargs.get('chol',not self.spdapx)
            if chol:
                try:
                    if isinstance(self.s,int):
                        L,P=sparse_cholesky(self.prec())
                        cholinvC=P.dot(L)**self.s/np.sqrt(self.sigma2)
                        y=multf(cholinvC,x,transp) if alpha<0 else mdivf(cholinvC,x,transp)
                    else:
                        cholC=spla.cholesky(self.tomat(),lower=True)
                        y=multf(cholC,x,transp) if alpha>0 else mdivf(cholC,x,transp)
                except Exception as e:#spla.LinAlgError:
                    warnings.warn('Cholesky decomposition failed: '+str(e))
                    chol=False; pass
            if not chol:
                eigv,eigf=self.eigs(**kwargs)
                if alpha<0: eigv[abs(eigv)<np.finfo(float).eps]=np.finfo(float).eps
                y=multf(eigf*pow(eigv,alpha),multf(eigf.T,x,transp),transp)
        return y
    
    def update(self,sigma2=None,tau2=None,l=None):
        """
        Update the eigen-basis
        """
        if sigma2 is not None:
            sigma2_=self.sigma2
            self.sigma2=sigma2
            if self.store_eig:
                self.eigv*=self.sigma2/sigma2_
        if tau2 is not None:
            tau2_=self.tau2
            self.tau2=tau2
            if self.store_eig:
                self.eigv=pow(pow(self.eigv,-1./self.s)+(-tau2_+self.tau2)*self.sigma2**(-1./self.s),-self.s)
        if l is not None:
            self.l=l
            self.Lap=self._get_Laplacian(self.g, self.parameters.get('weightedge',False), self.parameters.get('normalize',False))
            if self.store_eig:
                self.eigv,self.eigf=self.eigs(upd=True)
        return self

if __name__=='__main__':
    np.random.seed(2022)
    
    import time
    t0=time.time()
    
    x='./satellite.png'
    ker=Ker(x,L=10,store_eig=True,normalize=False)
    verbose=ker.comm.rank==0 if ker.comm is not None else True
    if verbose:
        print('Eigenvalues :', np.round(ker.eigv[:min(10,ker.L)],4))
        print('Eigenvectors :', np.round(ker.eigf[:,:min(10,ker.L)],4))
    
    t1=time.time()
    if verbose:
        print('time: %.5f'% (t1-t0))
    
    ldet=ker.ldet()
    
    v=np.random.randn(ker.N)
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