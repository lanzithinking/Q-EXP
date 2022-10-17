"""
Plot prior samples
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the priors
from prior import prior


seed=2022
np.random.seed(seed)
# define the inverse problem
input = 128
ker_opt = 'serexp'
basis_opt = 'Fourier'
KL_trunc = 2000
space = 'fun' if ker_opt=='graphL' else 'vec'
sigma2 = 1
s = 1
# l = 10 
q = 1
store_eig = (ker_opt!='graphL')
prior_params={'input':input,
              'ker_opt':ker_opt,
              'basis_opt':basis_opt, # serexp param
              'KL_trunc':KL_trunc,
              'space':space,
              'sigma2':sigma2,
              's':s,
              # 'l':l,
              'q':q,
              'store_eig':store_eig}

# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)

plt.rcParams['image.cmap'] = 'binary'
# plot prior samples
fig,axes = plt.subplots(nrows=1,ncols=num_mdls,sharex=True,sharey=True,figsize=(12,4))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    prior_params['prior_option']=pri_mdls[i]
    prior_params['q']=2 if prior_params['prior_option']=='gp' else q
    pri = prior(**prior_params, seed=seed)
    u=pri.sample()
    if u.shape[0]!=pri.ker.N: u=pri.vec2fun(u)
    plt.axes(ax)
    img=u.reshape(pri.imsz)
    sub_figs[i]=plt.imshow(img, origin='lower',extent=[0, 1, 0, 1])
    ax.set_title(mdl_names[i],fontsize=16)
    ax.set_aspect('auto')
# set color bar
# from util.common_colorbar import common_colorbar
# fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig('./properties/prior_samples.png',bbox_inches='tight')
# plt.show()