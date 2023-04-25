#!/usr/bin/env python
"""
Class definition of data-misfit for time series.
----------------------------------------------------------------------------
Created October 4, 2022 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The Q-EXP project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import os

class misfit(object):
    """
    Class definition of data-misfit function
    """
    def __init__(self, **kwargs):
        """
        Initialize data-misfit class with information of observations.
        """
        self.truth_option = kwargs.pop('truth_option',0)
        self.truth_name = {0:'step',1:'turning'}[self.truth_option]
        self.size = kwargs.pop('size',200)
        self.times = np.linspace(0,2,self.size)
        self.truth = self._truef(opt=self.truth_option)(self.times)
        # self.nzlvl = kwargs.pop('nzlvl',0.01) # noise level
        self.nzlvl = kwargs.pop('nzlvl',0.015 if self.truth_option==1 else np.hstack([np.repeat(.01, int(self.size/2)), np.repeat(.007, int(self.size/4)), np.repeat(.007, int(self.size/4))])) # noise level
        # get observations
        self.obs, self.nzvar = self.get_obs(**kwargs)
    
    def _truef(self, opt=None):
        """
        Truth process
        """
        if opt is None: opt=self.truth_option
        if opt==0:
            f=lambda ts: np.array([1*(t>=0 and t<=1) + 0.5*(t>1 and t<=1.5) + 2*(t>1.5 and t<=2) for t in ts])
        elif opt==1:
            f=lambda ts: np.array([1.5*t*(t>=0 and t<=1) + (3.5-2*t)*(t>1 and t<=1.5) + (3*t-4)*(t>1.5 and t<=2) for t in ts])
        else:
            raise ValueError('Wrong option for truth!')
        return f
    
    def observe(self, opt=None):
        """
        Observe time series by adding noise
        """
        if opt is None: opt=self.truth_option
        obs_ts = self._truef(opt)(self.times)
        nzstd = self.nzlvl*np.linalg.norm(obs_ts)
        return obs_ts, nzstd**2
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_opt = kwargs.pop('opt',self.truth_option)
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='timeseries_obs_'+{0:'step',1:'turning'}[obs_opt]
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzvar=loaded['nzvar']
            print('Observation file '+obs_file_name+' has been read!')
            if obs.size!=self.size:
                raise ValueError('Stored observations not match the requested size! Regenerating...')
        except Exception as e:
            print(e); pass
            obs, nzvar = self.observe(opt=obs_opt)
            obs += np.sqrt(nzvar) * np.random.RandomState(kwargs.pop('rand_seed',2022)).randn(*obs.shape)
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzvar=nzvar)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzvar
    
    def cost(self, u):
        """
        Evaluate misfit function for given image (vector) u.
        """
        obs = u
        dif_obs = obs-self.obs
        val = 0.5*np.sum(dif_obs**2/self.nzvar)
        return val
    
    def grad(self, u):
        """
        Compute the gradient of misfit
        """
        obs = u
        dif_obs = obs-self.obs
        g = dif_obs/self.nzvar
        return g
    
    def Hess(self, u=None):
        """
        Compute the Hessian action of misfit
        """
        def hess(v):
            if v.ndim==1 or v.shape[0]!=self.size: v=v.reshape((self.size,-1))
            dif_obs = v
            Hv = dif_obs/(self.nzvar if self.nzvar.size==1 else self.nzvar[:,None])
            return Hv.squeeze()
        return hess
    
    def plot_data(self, dat=None, save_plt=False, save_path='./reconstruction', **kwargs):
        """
        Plot the data information.
        """
        truths = kwargs.pop('truths', {0:self._truef(0)(self.times),1:self._truef(1)(self.times)})
        scatpts = kwargs.pop('scatpts', {0:self.get_obs(opt=0)[0],1:self.get_obs(opt=1)[0]})
        titles = kwargs.pop('titles', {0:'step',1:'turning'})
        n_dat = len(truths)
        import matplotlib.pyplot as plt
        plt.set_cmap(kwargs.pop('cmap','Greys'))
        # from util import matplot4dolfin
        # matplot=matplot4dolfin()
        fig, axes = plt.subplots(1, n_dat, sharex=True, sharey=True, figsize=(n_dat*5, 5))
        for i,ax in enumerate(axes.flat):
            plt.axes(ax)
            ax.plot(self.times, truths[i], linewidth=2)
            if dat is not None: ax.plot(self.times, dat[i], **kwargs)
            ax.scatter(self.times, scatpts[i], color='orange')
            ax.set_title(titles[i],fontsize=16)
            ax.set_aspect('auto')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if save_plt:
            if not os.path.exists(save_path): os.makedirs(save_path)
            save_fname = kwargs.pop('save_fname','-'.join(list(titles.values())))+'.png'
            plt.savefig(save_path+'/'+save_fname,bbox_inches='tight')
        return fig
    
if __name__ == '__main__':
    np.random.seed(2022)
    # define the misfit
    msft = misfit(truth_option=0, size=200)
    # msft = misfit(truth_option=1, size=200)
    # test
    u = msft.truth
    nll=msft.cost(u)
    grad=msft.grad(u)
    hess=msft.Hess(u)
    v = np.random.randn(*u.shape)
    h=1e-6
    gradv_fd=(msft.cost(u+h*v)-nll)/h
    gradv=grad.dot(v.flatten())
    rdiff_gradv=np.abs(gradv_fd-gradv)/np.linalg.norm(u)
    print('Relative difference of gradients in a direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    hessv_fd=(msft.grad(u+h*v)-grad)/h
    hessv=hess(v)
    rdiff_hessv=np.linalg.norm(hessv_fd-hessv)/np.linalg.norm(v)
    print('Relative difference of Hessian-action in a direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
    # plot
    import matplotlib.pyplot as plt
    fig=msft.plot_data(save_plt=True, save_path='./properties', save_fname='truth_obs.png')
    # plt.show()