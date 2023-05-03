#!/usr/bin/env python
"""
Class definition of data-misfit for linear inverse problem of Shepp-Logan head phantom.
---------------------------------------------------------------------------------------
Created April 26, 2023 for project of q-exponential process prior (Q-EXP)
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The Q-EXP project"
__credits__ = "Johnathan M. Bardsley"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com"

import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.io as spio
import scipy.sparse.linalg as spsla
from scipy.ndimage import convolve

import os,sys
sys.path.append( "../" )

class misfit(object):
    """
    Class definition of data-misfit function
    """
    def __init__(self, **kwargs):
        """
        Initialize data-misfit class with information of observations.
        """
        self.CT_set = kwargs.pop('CT_set','proj60_loc100') # CT set
        self.SNR = kwargs.pop('SNR',100)
        # get observations
        self.proj, self.obs, self.nzvar, self.size, self.truth = self.get_obs(**kwargs)
    
    def _gen_shepp_logan(self):
        """
        Generate Shepp Logan head phantom observations
        """
        CT_name = 'CT_x128_'+self.CT_set
        dat_name = 'Shepp-Logan_'+self.CT_set.split('_')[0]+'_SNR'+str(self.SNR)
        CT = spio.loadmat('./'+CT_name+'.mat')
        A,phi,s = CT['A'],CT['phi'],CT['s']
        data = spio.loadmat('./data/'+dat_name+'.mat')
        x_true,b = data['x_true'], data['b']
        return A, phi, s, x_true, b
    
    def observe(self):
        """
        Observe image projections
        """
        projection, angles, distances, truth, observation = self._gen_shepp_logan()
        observation = observation.flatten(order='F')
        noise = projection.dot(truth.flatten(order='F'))-observation
        nzvar = np.var(noise)
        size = truth.shape
        return projection, observation, nzvar, size, truth
    
    def get_obs(self, **kwargs):
        """
        Get observations
        """
        obs_file_loc=kwargs.pop('obs_file_loc',os.getcwd())
        obs_file_name='CT_obs_'+self.CT_set.split('_')[0]
        try:
            loaded=np.load(os.path.join(obs_file_loc,obs_file_name+'.npz'),allow_pickle=True)
            proj=loaded['proj'][0]; obs=loaded['obs']; nzvar=loaded['nzvar']; size=loaded['size']; truth=loaded['truth']
            print('Observation file '+obs_file_name+' has been read!')
        except Exception as e:
            print(e); pass
            proj, obs, nzvar, size, truth = self.observe()
            save_obs=kwargs.pop('save_obs',True)
            if save_obs:
                np.savez_compressed(os.path.join(obs_file_loc,obs_file_name), proj=(proj,), obs=obs, nzvar=nzvar, size=size, truth=truth)
            print('Observation'+(' file '+obs_file_name if save_obs else '')+' has been generated!')    
        return proj, obs, nzvar, size, truth
    
    def cost(self, u):
        """
        Evaluate misfit function for given image (vector) u.
        """
        A, b = self.proj, self.obs
        dif_obs = A.dot(u)-b
        val = 0.5*np.sum(dif_obs**2)/self.nzvar
        return val
    
    def grad(self, u):
        """
        Compute the gradient of misfit
        """
        A, b = self.proj, self.obs
        dif_obs = A.dot(u)-b
        g = A.T.dot(dif_obs).flatten()/self.nzvar
        return g
    
    def Hess(self, u=None):
        """
        Compute the Hessian action of misfit
        """
        A = self.proj
        def hess(v):
            if v.ndim==1 or v.shape[0]!=np.prod(self.size): v=v.reshape((np.prod(self.size),-1),order='F') # cautious about multiple images: can be [n_imgs, H, W]
            Hv = np.stack([A.T.dot(A.dot(v[:,k])).flatten()/self.nzvar for k in range(v.shape[-1])]).T
            return Hv.squeeze()
        return hess
    
    def reconstruct_LSE(self,lmda=0.1):
        """
        Reconstruct images by least square estimate
        """
        A, b = self.proj, self.obs
        rcstr = spsla.lsqr(A, b=b, damp=lmda)[0]
        return rcstr.reshape(self.size, order='F')
    
    def plot_data(self, img=None, save_img=False, save_path='./reconstruction', **kwargs):
        """
        Plot the data information.
        """
        images = kwargs.pop('images', {0:self.truth,1:img})
        titles = kwargs.pop('titles', {0:'Truth',1:'Reconstruction'})
        n_imgs = len(images)
        import matplotlib.pyplot as plt
        plt.set_cmap(kwargs.pop('cmap','gray'))#'Greys'))
        # from util import matplot4dolfin
        # matplot=matplot4dolfin()
        fig, axes = plt.subplots(1, n_imgs, sharex=True, sharey=True, figsize=(n_imgs*5, 5))
        for i,ax in enumerate(axes.flat):
            plt.axes(ax)
            ax.imshow(images[i],extent=[0, 1, 0, 1])
            ax.set_title(titles[i],fontsize=16)
            ax.set_aspect('auto')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if save_img:
            if not os.path.exists(save_path): os.makedirs(save_path)
            save_fname = kwargs.pop('save_fname',titles[n_imgs-1])+'.png'
            plt.savefig(save_path+'/'+save_fname,bbox_inches='tight')
        return fig
    
if __name__ == '__main__':
    np.random.seed(2022)
    # define the misfit
    msft = misfit(CT_set='proj90_loc100')
    # test
    u = np.random.randn(np.prod(msft.size))
    nll=msft.cost(u)
    grad=msft.grad(u)
    hess=msft.Hess(u)
    h=1e-6; v=msft.truth.flatten(order='F')
    gradv_fd=(msft.cost(u+h*v)-nll)/h
    gradv=grad.dot(v)
    rdiff_gradv=np.abs(gradv_fd-gradv)/np.linalg.norm(v)
    print('Relative difference of gradients in a direction between direct calculation and finite difference: %.10f' % rdiff_gradv)
    hessv_fd=(msft.grad(u+h*v)-grad)/h
    hessv=hess(v)
    rdiff_hessv=np.linalg.norm(hessv_fd-hessv)/np.linalg.norm(v)
    print('Relative difference of Hessian-action in a direction between direct calculation and finite difference: %.10f' % rdiff_hessv)
    # plot
    import matplotlib.pyplot as plt
    # fig=msft.plot_data()
    # # fig.tight_layout()
    # fig.savefig('./properties/truth_obs.png',bbox_inches='tight')
    # plt.show()
    # plot reconstruction
    projection, angles, distances, truth, observation = msft._gen_shepp_logan()
    rcstr_LSE=msft.reconstruct_LSE(lmda=0.1)
    fig=msft.plot_data(images={0:msft.truth,1:msft.obs.reshape((angles.size,distances.size),order='F').T,2:rcstr_LSE}, titles={0:'Truth',1:'Observation',2:'LSE-reconstruction'},
                       save_img=True, save_path='./properties', save_fname='truth_obs_recnstr', cmap='gray')
    plt.show()