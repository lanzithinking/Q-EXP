#!/usr/bin/env python
"""
Class definition of data-misfit for time series.
----------------------------------------------------------------------------
Created October 4, 2022 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shuyi Li"
__copyright__ = "Copyright 2022, The Q-EXP project"
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import os
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class misfit(object):
    """
    Class definition of data-misfit function
    """
    def __init__(self, **kwargs):
        """
        Initialize data-misfit class with information of observations.
        """
        self.data_option = kwargs.pop('data_option',2)
        self.data_name = {0:'total',1:'100', 2:'50'}[self.data_option]
        self.size = kwargs.pop('size',229)
        self.diff = kwargs.pop('diff',0)
        now = dt.datetime.strptime('01/21/2020','%m/%d/%Y').date() #dt.datetime.now()
        then = now + dt.timedelta(days=self.size) 
        days = mdates.drange(now,then,dt.timedelta(days=1 if not self.diff else 2))
        self.times = days#np.linspace(0,2,self.size)
        self.size = len(self.times)
        self.nzlvl = kwargs.pop('nzlvl',0.001 if self.diff else 0.005 ) # noise level
        # get observations
        self.obs, self.nzvar = self.get_obs(**kwargs)
    
    def _data(self, opt=None):
        """
        Obtain data
        """
        from scipy.io import loadmat
        
        if opt is None: opt=self.data_option
        covid = loadmat('covid.mat')
        case = covid['y'].squeeze()
        if self.diff:
            case_diff = np.diff(case,axis=1)[5] #56,228->1,114 for CA take every other day
            dat = case_diff[::2] if opt==0 else case_diff[:100][::2] #only 0/1  
        else:
            casesum = np.sum(case,axis=0)        
            dat = casesum if opt==0 else casesum[:100] if opt==1 else casesum[:50] 
        return dat
    
    def observe(self, opt=None):
        """
        Observe time series by adding noise
        """

        if opt is None: opt=self.data_option
        obs_ts = self._data(opt)
        nzstd = self.nzlvl*np.std(obs_ts)
        return obs_ts, nzstd**2
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_opt = kwargs.pop('opt',self.data_option)
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='covid_CA_obs_'+{0:'total',1:'100', 2:'50'}[obs_opt] if self.diff else f'covid_nl{self.nzlvl}_obs_'+{0:'total',1:'100', 2:'50'}[obs_opt]
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzvar=loaded['nzvar']
            print('Observation file '+obs_file_name+' has been read!')
            if obs.size!=self.size:
                raise ValueError('Stored observations not match the requested size! Regenerating...')
        except Exception as e:
            print(e); pass
            obs, nzvar = self.observe(opt=obs_opt)
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
    
    def plot_data(self):
        """
        Plot the data information.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        #plt.set_cmap('Greys')
        days = self.times
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.plot(days, self.obs)
        # ax.scatter(days, self.obs, color='orange', s=15)
        plt.gcf().autofmt_xdate()
        # plt.show()
        return fig
    
if __name__ == '__main__':
    np.random.seed(2022)
    # define the misfit, size=229 for opt=0, 100 for opt=1, 50 for opt=2
    opt = 0
    size = 229 if opt==0 else 100 if opt==1 else 50
    msft = misfit(data_option=opt, size=size)
    #msft = misfit(data_option=1, size=200)
    # test
    u = msft.obs
    nll=msft.cost(u)
    grad=msft.grad(u)
    v = np.random.randn(*u.shape)
    h=1e-6
    gradv_fd=(msft.cost(u+h*v)-nll)/h
    gradv=grad.dot(v.flatten())
    rdiff_gradv=np.abs(gradv_fd-gradv)/np.linalg.norm(u)
    print('Relative difference of gradients in a direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    # plot
    fig=msft.plot_data()
    # fig.tight_layout()
    savepath='./properties'
    if not os.path.exists(savepath): os.makedirs(savepath)
    fig.savefig(os.path.join(savepath,'data_obs.png'),bbox_inches='tight')
    # plt.show()