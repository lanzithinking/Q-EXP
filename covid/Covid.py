#!/usr/bin/env python
"""
Class definition of the time series problem
Shuyi Li @ ASU 2022
----------------------------------------------------------------------------
Created January 20, 2023 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The Q-EXP project"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
from scipy import optimize
import os

# self defined modules
from prior import *
from misfit import *

# set to warn only once for the same warnings
import warnings
warnings.simplefilter('once')

class Covid:
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
        self.prior = prior(input=self.misfit.size,L=self.KL_trunc,**kwargs)
        print('\nPrior model is specified.')
        # set low-rank approximate Gaussian posterior
        # self.post_Ga = Gaussian_apx_posterior(self.prior,eigs='hold')
        # print('\nApproximate posterior model is set.\n')
    
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
        raise NotImplementedError('HessApply not implemented.')
    
    def get_geom(self,parameter=None,geom_ord=[0],**kwargs):
        """
        Get necessary geometric quantities including log-likelihood (0), adjusted gradient (1), 
        Hessian apply (1.5), and its eigen-decomposition using randomized algorithm (2).
        """
        if parameter is None:
            parameter=self.prior.mean
        loglik=None; agrad=None; HessApply=None; eigs=None;
        
        # # convert parameter vector to function
        # parameter = self.prior.vec2fun(parameter)
        
        # get log-likelihood
        if any(s>=0 for s in geom_ord):
            loglik = -self._get_misfit(parameter, **kwargs)
        
        # get gradient
        if any(s>=1 for s in geom_ord):
            agrad = -self._get_grad(parameter, **kwargs)
        
        # get Hessian Apply
        if any(s>=1.5 for s in geom_ord):
            pass
        
        # get estimated eigen-decomposition for the Hessian (or Gauss-Newton)
        if any(s>1 for s in geom_ord):
            pass
        
        return loglik,agrad,HessApply,eigs
    
    def get_eigs(self,parameter=None,**kwargs):
        """
        Get the eigen-decomposition of Hessian action directly using randomized algorithm.
        """
        raise NotImplementedError('eigs not implemented.')
    
    def get_MAP(self,SAVE=False,**kwargs):
        """
        Get the maximum a posterior (MAP).
        """
        import time
        sep = "\n"+"#"*80+"\n"
        print( sep, "Find the MAP point", sep)
        # set up initial point
        param0 = self.prior.sample()
        # param0 = self.misfit.obs.flatten()
        if self.prior.space=='vec': param0=self.prior.fun2vec(param0)
        fun = lambda parameter: self._get_misfit(parameter, MF_only=False)
        grad = lambda parameter: self._get_grad(parameter, MF_only=False)
        global Nfeval
        Nfeval=1
        def call_back(Xi):
            global Nfeval
            print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], fun(Xi)))
            Nfeval += 1
        print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
        # solve for MAP
        start = time.time()
        res = optimize.minimize(fun, param0, method='BFGS', jac=grad, callback=call_back, options={'maxiter':1000,'disp':True})
        # res = optimize.minimize(fun, param0, method='L-BFGS-B', jac=grad, callback=call_back, options={'maxiter':1000,'disp':True})
        # res = optimize.minimize(fun, param0, method='Newton-CG', jac=grad, callback=call_back, options={'maxiter':500,'disp':True})
        end = time.time()
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
            save_name=kwargs.pop('save_name','MAP')
            f = open(os.path.join(fld_name,save_name+'.pckl'),'wb')
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
    
    def test(self,h=1e-4):
        """
        Demo to check results with the adjoint method against the finite difference method.
        """
        # random sample parameter
        parameter = self.prior.sample()
        
        # MF_only = True
        import time
        # obtain the geometric quantities
        print('\n\nObtaining geometric quantities by direct calculation...')
        start = time.time()
        loglik,grad,_,_ = self.get_geom(parameter,geom_ord=[0,1])
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
        loglik_p = -self._get_misfit(parameter_p)
#         parameter_m = parameter - h*v
#         loglik_m = -self._get_misfit(parameter_m)
        dloglikv_fd = (loglik_p-loglik)/h
        dloglikv = grad.dot(v.flatten())
        rdiff_gradv = np.abs(dloglikv_fd-dloglikv)/np.linalg.norm(v)
        print('Relative difference of gradients in a random direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
        end = time.time()
        print('Time used is %.4f' % (end-start))
    
if __name__ == '__main__':
    # set up random seed
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    seed, opt, diff=2022, 0, 0 #0-opt, 1-diff or  2-opt, 0-diff
    nzlvl={0:0.001,2:0.005}[opt]
    np.random.seed(seed)
    if not diff:
        size = 229 if opt==0 else 100 if opt==1 else 50
    else:
        size = 228 if opt==0 else 100
    
    map_fl,relerr_l = [],[]
    for prior_name in ['gp','bsv','qep']:
        # define Bayesian inverse problem
        prior_params={'prior_option':prior_name,
                      'ker_opt':'covf',
                      'cov_opt':'matern',
                      'basis_opt':'Fourier', # serexp param
                      'KL_trunc':100,
                      'space':'fun',
                      'sigma2':{0:100,2:1}[opt],
                      's':1,
                      'q':2 if prior_name=='gp' else 1,
                      'store_eig':True}
        lik_params={'data_option':opt,
                    'size':size,
                    'diff':diff,
                    'nzlvl':nzlvl}
        
        # if prior_name=='gp' and opt==0:
        #     prior_params['sigma2'] = 1000
        ts = Covid(**prior_params,**lik_params,seed=seed)
        # test
        ts.test(1e-8)
        # obtain MAP
        map = ts.get_MAP(SAVE=True, save_name='MAP_'+ts.misfit.data_name+'_'+prior_params['prior_option'])
        print('MAP estimate: '+(min(len(map),10)*"%.4f ") % tuple(map[:min(len(map),10)]) )
        #  compare it with the data
        dat = ts.misfit.obs
        map_f = ts.prior.vec2fun(map) if ts.prior.space=='vec' else map
        map_f = map_f.reshape(dat.shape)
        map_fl.append(map_f)
        relerr = np.linalg.norm(map_f-dat)/np.linalg.norm(dat)
        relerr_l.append(relerr)
        print('Relative RMSE of MAP compared with the data %.2f%%' % (relerr*100))
    
        # plot single MAP
        fig,ax = plt.subplots()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        ax.plot(ts.misfit.times, map_f, linewidth=2, linestyle='--', color='blue', label = 'MAP')
        ax.scatter(ts.misfit.times, ts.misfit.obs, color='orange', label='obs', s=15)
        plt.legend()
        plt.title('MAP ('+ts.misfit.data_name+f') for %s' %(prior_params['prior_option']),fontsize=16)
        savepath='./properties'
        if not os.path.exists(savepath): os.makedirs(savepath)
        plt.savefig(os.path.join(savepath,'MAP_'+ts.misfit.data_name+'_'+prior_params['prior_option']+'.png'),bbox_inches='tight')
        # plt.show()
    
    
    # plot MAP comparison
    num_rows=1
    mdl_names=['Gaussian','Besov','q-Exponential']
    num_mdls=len(mdl_names)

    # posterior median
    fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls,sharex=True,sharey=True,figsize=(16,5))
    titles = mdl_names
    for i,ax in enumerate(axes.flat):
        plt.axes(ax)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.plot(ts.misfit.times, map_fl[i], linewidth=2, linestyle='--', color='blue')
        ax.scatter(ts.misfit.times, ts.misfit.obs, color='orange', s=10)
        plt.gcf().autofmt_xdate()
        ax.set_title(titles[i],fontsize=18)
        ax.set_aspect('auto')
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # save plot
    # fig.tight_layout()
    folder = './MAP'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder,f'Covid_nl{nzlvl}_'+ts.misfit.data_name+'_maps_comparepri.png'),bbox_inches='tight')
    # plt.show()
