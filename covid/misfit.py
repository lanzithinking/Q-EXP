#!/usr/bin/env python
"""
Class definition of data-misfit for time series.
----------------------------------------------------------------------------
Created October 4, 2022 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The Q-EXP project"
__credits__ = "Mirjeta Pasha"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import os
import datetime as dt
import matplotlib.dates as mdates

class misfit(object):
    """
    Class definition of data-misfit function
    """
    def __init__(self, **kwargs):
        """
        Initialize data-misfit class with information of observations.
        """
        self.truth_option = kwargs.pop('truth_option',2)
        self.truth_name = {0:'total',1:'100', 2:'50'}[self.truth_option]
        self.size = kwargs.pop('size',229)
        self.diff = kwargs.pop('diff',0)
        now = dt.datetime.strptime('01/21/2020','%m/%d/%Y').date() #dt.datetime.now()
        then = now + dt.timedelta(days=self.size) 
        days = mdates.drange(now,then,dt.timedelta(days=1 if not self.diff else 2))
        self.times = days#np.linspace(0,2,self.size)
        self.truth = self._truef(opt=self.truth_option)
        self.size = len(self.times)
        #self.nzlvl = kwargs.pop('nzlvl',0.001) # noise level 0.000001
        #if self.truth_option==0 else np.hstack([np.repeat(.01, int(self.size/2)), np.repeat(.007, int(self.size/4)), np.repeat(.007, int(self.size/4))])
        self.nzlvl = kwargs.pop('nzlvl',0.001 if self.diff else 0.005 ) # noise level
        # get observations
        self.obs, self.nzvar = self.get_obs(**kwargs)
    
    def _truef(self, opt=None):
        """
        Truth process
        """
        from scipy.io import loadmat
        
        if opt is None: opt=self.truth_option
        covid = loadmat('covid.mat')
        case = covid['y'].squeeze()
        if self.diff:
            case_diff = np.diff(case,axis=1)[5] #56,228->1,114 for CA take every other day
            f = case_diff[::2] if opt==0 else case_diff[:100][::2] #only 0/1  
        else:
            casesum = np.sum(case,axis=0)        
            f = casesum if opt==0 else casesum[:100] if opt==1 else casesum[:50] 
        return f
    
    def observe(self, opt=None):
        """
        Observe time series by adding noise
        """

        if opt is None: opt=self.truth_option
        obs_ts = self._truef(opt)
        nzstd = self.nzlvl*np.linalg.norm(obs_ts)
        return obs_ts, nzstd**2
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_opt = kwargs.pop('opt',self.truth_option)
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
            print(obs.shape)
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
    
    def plot_data(self):
        """
        Plot the data information.
        """
        import matplotlib.pyplot as plt
        #plt.set_cmap('Greys')
        # from util import matplot4dolfin
        # matplot=matplot4dolfin()
        
        days = self.times

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.plot(days, self.truth) #
        plt.scatter(days, msft.obs, color='orange')
        plt.gcf().autofmt_xdate()
        plt.show()
        '''
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
        for i,ax in enumerate(axes.flat):
            f_i = self._truef(opt=i)
            plt.axes(ax)
            plt.plot(self.times, f_i(self.times), linewidth=2)
            plt.scatter(self.times, self.get_obs(opt=i)[0], color='orange')
            ax.set_title({0:'step',1:'turning'}[i]+' function',fontsize=16)
            ax.set_aspect('auto')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        '''
        return fig
    
if __name__ == '__main__':
    np.random.seed(2022)
    # define the misfit, size=229 for opt=0, 100 for opt=1, 50 for opt=2
    opt = 2
    size = 229 if opt==0 else 100 if opt==1 else 50
    msft = misfit(truth_option=opt, size=size)
    #msft = misfit(truth_option=1, size=200)
    # test
    u = msft.truth
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
    fig.savefig(os.path.join(savepath,'truth_obs.png'),bbox_inches='tight')
    # plt.show()
    
    '''
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    
    covid = loadmat('covid.mat')
    case = covid['y']
    case = case.squeeze()
    casesum = np.sum(case,axis=0)
    states = pd.read_csv('states.csv')
    a = list(states['state'])
    state_name = a[:2]+['?']+a[2:] #CA(5), FL(10), IL(15), LA(20), MI(24),MO(27)
    case_diff = np.diff(case,axis=1)
    plt.figure(figsize=(4,4))
    #plt.plot(np.arange(len(casesum))[:50],casesum[:50], label='total case')
    for i in range(5,6):
        plt.plot(np.arange(case_diff.shape[1])[::2],case_diff[i,::2], label=state_name[i])
    
    plt.legend()
    plt.xlabel(r'time')
    plt.ylabel('infected case')
    plt.title(r'total case in us in first 50days' ) #Likelihood: exp(-$l^{\alpha}$)
        
    
    '''