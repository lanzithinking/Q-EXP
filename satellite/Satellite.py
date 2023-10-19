#!/usr/bin/env python
"""
Class definition of linear inverse problem
----------------------------------------------------------------------------
For submission 'Bayesian Learning via Q-Exponential Process' to NIPS 2023
"""

import numpy as np
import scipy.sparse.linalg as spsla
from scipy import optimize
import os

# self defined modules
# from prior_bsv import *
# from prior_qep import *
from prior import *
from misfit import *
# from posterior_lr import *
# from whiten import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class Satellite:
    def __init__(self,**kwargs):
        """
        Initialize the linear inverse problem by defining the prior model and the misfit (likelihood) model.
        """
        self.KL_trunc=kwargs.pop('KL_trunc', 100)
        
        # define the inverse problem with prior, and misfit
        seed = kwargs.pop('seed',2022)
        self.setup(seed,**kwargs)
    
    def setup(self, seed=2022, **kwargs):
        """
        Set up the prior and the likelihood (misfit: -log(likelihood)) and posterior
        """
        # set (common) random seed
        np.random.seed(seed)
        sep = "\n"+"#"*80+"\n"
        # set misfit
        self.misfit = misfit(**kwargs)
        print('\nLikelihood model is obtained.')
        # set prior
        self.prior = prior(input=self.misfit.obs if kwargs.get('ker_opt','serexp')=='graphL' else self.misfit.size,L=self.KL_trunc,**kwargs)
        # self.whiten = whiten(self.prior)
        print('\nPrior model is specified.')
        # set low-rank approximate Gaussian posterior
        # self.post_Ga = Gaussian_apx_posterior(self.prior,eigs='hold')
        # print('\nApproximate posterior model is set.\n')
        if kwargs.pop('init_param',False):
            # obtain an initial parameter from a coarse reconstruction
            self._init_param()
        # if self.prior.mean is None: self.prior.mean = self.init_parameter
    
    def _init_param(self,init_opt='obs',**kwargs):
        """
        Initialize parameter with a quick but rough reconstruction
        """
        reconstruction_method='reconstruct_'+init_opt
        if hasattr(self.misfit, reconstruction_method):
            reconstruct=getattr(self.misfit,reconstruction_method)
            self.init_parameter = reconstruct(**kwargs).flatten()
        else:
            self.init_parameter = self.misfit.obs.flatten()
        if self.prior.space=='vec': self.init_parameter = self.prior.fun2vec(self.init_parameter)
        return self.init_parameter
    
    def _get_misfit(self, parameter, MF_only=True, **kwargs):
        """
        Compute the misfit (default), or the negative log-posterior for given parameter.
        """
        # evaluate data-misfit function
        msft = self.misfit.cost(self.prior.vec2fun(parameter) if self.prior.space=='vec' else parameter)
        if not MF_only: msft += self.prior.cost(parameter, **kwargs)
        return msft
    
    def _get_grad(self, parameter, MF_only=True):
        """
        Compute the gradient of misfit (default), or the gradient of negative log-posterior for given parameter.
        """
        # obtain the gradient
        grad = self.misfit.grad(self.prior.vec2fun(parameter) if self.prior.space=='vec' else parameter)
        if self.prior.space=='vec': grad = self.prior.fun2vec(grad)
        if not MF_only: grad += self.prior.grad(parameter)
        return grad

    def _get_HessApply(self, parameter=None, MF_only=True):
        """
        Compute the Hessian apply (action) for given parameter,
        default to the Gauss-Newton approximation.
        """
         # obtain the Hessian
        hess_ = lambda v: self.prior.fun2vec(self.misfit.Hess()(self.prior.vec2fun(v))) if self.prior.space=='vec' else self.misfit.Hess()(v)
        hess = hess_ if MF_only else lambda v: hess_(v) + self.prior.Hess(parameter)(v)
        return hess
    
    def get_geom(self,parameter=None,geom_ord=[0],**kwargs):
        """
        Get necessary geometric quantities including log-likelihood (0), adjusted gradient (1), 
        Hessian apply (1.5), and its eigen-decomposition using randomized algorithm (2).
        """
        if parameter is None:
            parameter=self.prior.mean
        loglik=None; agrad=None; HessApply=None; eigs=None;
        # optional arguments
        whitened=kwargs.pop('whitened',False)
        MF_only=kwargs.pop('MF_only',True)
        
        # un-whiten if necessary
        param = self.whiten.wn2qep(parameter) if whitened else parameter
        
        # get log-likelihood
        if any(s>=0 for s in geom_ord):
            loglik = -self._get_misfit(param, **kwargs)
            if whitened: loglik += self.whiten.jacdet(parameter)
        
        # get gradient
        if any(s>=1 for s in geom_ord):
            agrad = -self._get_grad(param)
            if whitened: agrad = self.whiten.wn2qep(parameter,1)(agrad, adj=True) + self.whiten.jacdet(parameter, 1)
        
        # get Hessian Apply
        if any(s>=1.5 for s in geom_ord):
            HessApply_ = self._get_HessApply(param, MF_only=MF_only) # Hmisfit if MF is true
            Hess0_,invHess0_=self.prior.Hess(param),self.prior.invHess(param)
            if whitened:
                HessApply = lambda v: self.whiten.wn2qep(parameter,1)(HessApply_(self.whiten.wn2qep(parameter,1)(v)), adj=True) \
                            # + self.whiten.wn2qep(parameter, 2)(v, self._get_grad(param, MF_only=MF_only), adj=True) # exact Hessian
                Hess0 = lambda v: self.whiten.wn2qep(parameter, 1)(Hess0_(self.whiten.wn2qep(parameter, 1)(v)), adj=True) \
                            + self.whiten.wn2qep(parameter, 2)(v, self.prior.grad(param), adj=True)
                invHess0 = lambda v: self.whiten.qep2wn(param, 1)(invHess0_(self.whiten.qep2wn(param, 1)(v, adj=True))) #+ ? self.whiten.qep2wn(parameter, 2)(v, self.prior.grad(param)) 
            else:
                HessApply,Hess0,invHess0 = HessApply_,Hess0_,invHess0_
            # v=np.random.randn(self.prior.L*self.prior.J)
            # v1=invHess0(Hess0(v))
            # print('Relative difference of invHessian-Hessian-action in a random direction between the composition and identity: %.10f' % (np.linalg.norm(v1-v)/np.linalg.norm(v)))
            # v2=Hess0(invHess0(v))
            # print('Relative difference of Hessian-invHessian-action in a random direction between the composition and identity: %.10f' % (np.linalg.norm(v2-v)/np.linalg.norm(v)))
            if np.max(geom_ord)<=1.5:
                # adjust the gradient
                agrad += HessApply(parameter)
                if not MF_only: agrad -= Hess0(parameter)
        
        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        if any(s>1 for s in geom_ord):
            if MF_only:
                self.posterior = posterior(HessApply, Hess0, invHess0, N=parameter.size, store_eig=True, **kwargs)
            else:
                self.posterior = posterior(HessApply, N=parameter.size, store_eig=True, **kwargs)
            eigs = self.posterior.eigs(**kwargs)
            if any(s>1.5 for s in geom_ord):
                # adjust the gradient
                agrad += self.posterior.K_act(parameter, -1)
                if not MF_only: agrad -= Hess0(parameter)
        
        return loglik,agrad,HessApply,eigs
    
    def get_eigs(self,parameter=None,**kwargs):
        """
        Get the eigen-decomposition of Hessian action directly using randomized algorithm.
        """
        k=kwargs.pop('k',int(self.prior.L/2))
        maxiter=kwargs.pop('maxiter',100)
        tol=kwargs.pop('tol',1e-10)
        HessApply = self._get_HessApply(parameter, kwargs.pop('MF_only',True)) # Hmisfit if MF is true
        H_op = spsla.LinearOperator((parameter.size,)*2,HessApply)
        eigs = spsla.eigsh(H_op,min(k,H_op.shape[0]-1),maxiter=maxiter,tol=tol)
        return eigs
    
    def get_MAP(self,SAVE=False,PRINT=True,**kwargs):
        """
        Get the maximum a posterior (MAP).
        """
        ncg = kwargs.pop('NCG',False) # choose whether to use conjugate gradient optimization method
        import time
        sep = "\n"+"#"*80+"\n"
        if PRINT: print( sep, "Find the MAP point"+({True:' using Newton CG',False:''}[ncg]), sep)
        # set up initial point
        # param0 = self.prior.sample()
        param0 = kwargs.pop('param0', self.init_parameter if hasattr(self, 'init_parameter') else self._init_param(**kwargs))
        # if self.prior.space=='vec': param0=self.prior.fun2vec(param0)
        fun = lambda parameter: self._get_misfit(parameter, MF_only=False)
        grad = lambda parameter: self._get_grad(parameter, MF_only=False)
        if ncg: hessp = lambda parameter, v: self._get_HessApply(parameter, MF_only=False)(v)
        global Nfeval
        Nfeval=1
        def call_back(Xi):
            global Nfeval
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], fun(Xi)))
            Nfeval += 1
        if PRINT: print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
        # solve for MAP
        options = kwargs.pop('options',{'maxiter':1000,'disp':True})
        start = time.time()
        if ncg:
            # res = optimize.minimize(fun, param0, method='Newton-CG', jac=grad, hessp=hessp, callback=call_back if PRINT else None, options=options)
            res = optimize.minimize(fun, param0, method='trust-ncg', jac=grad, hessp=hessp, callback=call_back if PRINT else None, options=options)
        else:
            # res = optimize.minimize(fun, param0, method='BFGS', jac=grad, callback=call_back if PRINT else None, options=options)
            res = optimize.minimize(fun, param0, method='L-BFGS-B', jac=grad, callback=call_back if PRINT else None, options=options)
        end = time.time()
        if PRINT:
            print('\nTime used is %.4f' % (end-start))
            # print out info
            if res.success:
                print('\nConverged in ', res.nit, ' iterations.')
            else:
                print('\nNot Converged.')
            print('Final function value: %.4f.\n' % res.fun)
        
        MAP = res.x
        
        if SAVE:
            import pickle
            fld_name='properties'
            self._check_folder(fld_name)
            f = open(os.path.join(fld_name,'MAP.pckl'),'wb')
            pickle.dump(MAP, f)
            f.close()
        
        return MAP
    
    def _check_folder(self,fld_name='result'):
        """
        Check the existence of folder for storing result and create one if not
        """
        if not hasattr(self, 'savepath'):
            cwd=os.getcwd()
            self.savepath=os.path.join(cwd,fld_name)
        if not os.path.exists(self.savepath):
            print('Save path does not exist; created one.')
            os.makedirs(self.savepath)
    
    def test(self,h=1e-4,MF_only=True):
        """
        Demo to check results with the adjoint method against the finite difference method.
        """
        # random sample parameter
        # parameter = self.prior.sample()
        if not hasattr(self, 'init_parameter'): self._init_param()
        parameter = self.init_parameter
        
        # MF_only = True
        import time
        # obtain the geometric quantities
        print('\n\nObtaining geometric quantities by direct calculation...')
        start = time.time()
        # loglik,grad,HessApply,_ = self.get_geom(parameter,geom_ord=[0,1,1.5],MF_only=MF_only)
        loglik = -self._get_misfit(parameter,MF_only=MF_only)
        grad = -self._get_grad(parameter,MF_only=MF_only)
        HessApply = self._get_HessApply(parameter,MF_only=MF_only)
        end = time.time()
        print('Time used is %.4f' % (end-start))
        
        # check with finite difference
        print('\n\nTesting against Finite Difference method...')
        start = time.time()
        # random direction
        v = self.prior.sample()
        ## gradient
        print('\nChecking gradient:')
        parameter_p = parameter + h*v
        loglik_p = -self._get_misfit(parameter_p,MF_only=MF_only)
#         parameter_m = parameter - h*v
#         loglik_m = -self._get_misfit(parameter_m,MF_only=MF_only)
        dloglikv_fd = (loglik_p-loglik)/h
        dloglikv = grad.dot(v.flatten())
        rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/np.linalg.norm(v)
        print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
        ## Hessian
        print('\nChecking Hessian:')
        grad_p = -self._get_grad(parameter_p,MF_only=MF_only)
        dgradv_fd = -(grad_p-grad)/h
        dgradv = HessApply(v)
        rdiff_hessv = np.linalg.norm(dgradv_fd-dgradv)/np.linalg.norm(v)
        print('Relative difference of Hessian-action in a random direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
        end = time.time()
        print('Time used is %.4f' % (end-start))
    
if __name__ == '__main__':
    # set up random seed
    seed=2022
    np.random.seed(seed)
    # define Bayesian inverse problem
    prior_params={'prior_option':'qep', # change to alternatives 'gp', 'bsv'
                  'ker_opt':'serexp',
                  'basis_opt':'Fourier', # serexp param
                  'KL_trunc':2000,
                  'space':'vec',
                  'sigma2':1,
                  's':1,
                  'q':1,
                  'store_eig':True,
                  'normalize':True, # graphL param
                  'weightedge':True} # graphL param
    lik_params={'fltnz':2}
    # prior_option = 'gp'
    # fltnz = 2
    # basis_opt = 'Fourier'
    # KL_trunc = 2000
    # sigma2 = 1
    # s = 1
    # q = 1
    # store_eig = False
    satlit = Satellite(**prior_params,**lik_params, seed=seed)
    # test
    satlit.test(1e-8, MF_only=False)
    # obtain MAP
    map_v = satlit.get_MAP(SAVE=True, NCG=True)#,init_opt='LSE',lmda=.1)
    print('MAP estimate: '+(min(len(map_v),10)*"%.4f ") % tuple(map_v[:min(len(map_v),10)]) )
    #  compare it with the truth
    true_param = satlit.misfit.truth
    map_f = satlit.prior.vec2fun(map_v) if satlit.prior.space=='vec' else map_v
    map_f = map_f.reshape(true_param.shape)
    relerr = np.linalg.norm(map_f-true_param)/np.linalg.norm(true_param)
    print('Relative error of MAP compared with the truth %.2f%%' % (relerr*100))
    # report the minimum cost
    # min_cost = satlit._get_misfit(map_v)
    # print('Minimum cost: %.4f' % min_cost)
    # plot MAP
    import matplotlib.pyplot as plt
    msft=satlit.misfit
    fig=msft.plot_data(img=map_f, save_img=True, save_path='./properties')
    # fig.axes[1].set_title('MAP',fontsize=16)
    # fig.savefig('./properties/truth_map.png',bbox_inches='tight')
    plt.show()