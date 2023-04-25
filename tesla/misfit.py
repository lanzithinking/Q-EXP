#!/usr/bin/env python
"""
Class definition of data-misfit for time series.
----------------------------------------------------------------------------
Created October 4, 2022 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shuyi Li"
__copyright__ = "Copyright 2022, The Q-EXP project"
__license__ = "GPL"
__version__ = "0.4"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import pandas as pd
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
        self.start_date = kwargs.pop('start_date','2022-01-01')
        self.end_date = kwargs.pop('end_date','2023-01-01')
        if 'days' in kwargs:
            days=kwargs.pop('days',0)
            if days>=0:
                adj_end=dt.datetime.strptime(self.start_date,'%Y-%m-%d').date()+dt.timedelta(days=days)
                self.end_date=adj_end.strftime('%Y-%m-%d')
            else:
                adj_start=dt.datetime.strptime(self.end_date,'%Y-%m-%d').date()-dt.timedelta(days=abs(days))
                self.start_date=adj_start.strftime('%Y-%m-%d')
        self.stock_name = kwargs.pop('stock_name','TSLA')
        self.nzlvl = kwargs.pop('nzlvl', 0.1 ) # noise level
        # get observations
        self.obs, self.nzvar, self.times = self.get_obs(**kwargs)
        self.size = len(self.obs)
    
    def _get_data(self, **kwargs):
        """
        Obtain the stock data
        """
        dat_file_name=self.stock_name+'.csv'
        dat_file_loc=kwargs.pop('obs_loc',os.getcwd())
        if os.path.exists(os.path.join(dat_file_loc,dat_file_name)):
            stock=pd.read_csv(os.path.join(dat_file_loc,dat_file_name))
            print(dat_file_name+' has been read!')
        else:
            print('No local stock file found. Downloading...')
            try:
                import yfinance as yf
                stock=yf.download(self.stock_name, self.start_date, self.end_date)
                stock['Date']=pd.to_datetime(stock.index, format='%Y-%m-%d', utc=True)
                stock['Date']=stock['Date'].dt.date#strftime('%Y-%m-%d')
                stock=stock[['Date']+list(stock.columns[:-1].values)]
                stock.reset_index(drop=True, inplace=True)
                stock.to_csv(os.path.join(dat_file_loc,dat_file_name))
            except Exception as e:
                print(e)
                raise
        return stock
    
    def observe(self, **kwargs):
        """
        Observe time series by adding noise
        """
        stock = self._get_data(**kwargs)
        times = mdates.date2num(stock['Date'])
        obs_ts=stock['Close'].values
        nzstd = self.nzlvl*np.std(obs_ts)
        return obs_ts, nzstd**2, times
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name=self.stock_name+f'_nl{self.nzlvl}'
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            obs=loaded['obs']; nzvar=loaded['nzvar'];  times=loaded['times']
            print('Observation file '+obs_file_name+' has been read!')
        except Exception as e:
            print(e); pass
            obs, nzvar, times = self.observe(**kwargs)
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), obs=obs, nzvar=nzvar, times=times)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')
        return obs, nzvar, times
    
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
        if dat is None: dat = self.obs
        import matplotlib.pyplot as plt
        # plt.set_cmap(kwargs.pop('cmap','Greys'))
        fig, ax = plt.subplots(figsize=(12, 5))
        days = self.times
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.plot(days, self.obs) #
        # ax.scatter(days, self.obs, color='orange', s=15)
        plt.gcf().autofmt_xdate()
        # plt.show()
        if save_plt:
            if not os.path.exists(save_path): os.makedirs(save_path)
            save_fname = kwargs.pop('save_fname','Tesla_stocks')+'.png'
            plt.savefig(save_path+'/'+save_fname,bbox_inches='tight')
        return fig
    
if __name__ == '__main__':
    np.random.seed(2022)
    # define the misfit
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    msft = misfit(start_date=start_date, end_date=end_date)
    # test
    u = msft.obs
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
    fig=msft.plot_data(save_plt=True, save_path='./properties', save_fname='data_obs.png')
    # plt.show()