"""
Plot estimates of uncertainty field u in linear inverse problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from Linv import Linv


seed=2022
# define the inverse problem
fltnz = 2
ker_opt = 'serexp'
basis_opt = 'Fourier'
KL_truc = 2000
space = 'fun' if ker_opt=='graphL' else 'vec'
sigma2 = 1
s = 1
store_eig = (ker_opt!='graphL')
linv = Linv(fltnz=fltnz, ker_opt=ker_opt, basis_opt=basis_opt, KL_truc=KL_truc, space=space, sigma2=sigma2, s=s, store_eig=store_eig, seed=seed, normalize=True, weightedge=True)

# models
pri_mdls=('GP','BSV','qEP')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './analysis'
if os.path.exists(os.path.join(folder,'mcmc_summary.pckl')):
    f=open(os.path.join(folder,'mcmc_summary.pckl'),'rb')
    med_v,mean_v,std_v=pickle.load(f)
    f.close()
    print('mcmc_summary.pckl has been read!')
else:
    med_v=np.zeros((num_mdls,linv.prior.ker.N))
    mean_v=np.zeros((num_mdls,linv.prior.ker.N))
    std_v=np.zeros((num_mdls,linv.prior.ker.N))
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        fld_m = folder+'/'+pri_mdls[m]
        # preparation for estimates
        if os.path.exists(fld_m):
            pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
            for f_i in pckl_files:
                try:
                    f=open(os.path.join(fld_m,f_i),'rb')
                    f_read=pickle.load(f)
                    samp=f_read[-4]
                    if linv.prior.space=='vec': samp=linv.prior.vec2fun(samp.T).T
                    med_v[m]=np.median(samp,axis=0)
                    mean_v[m]=np.mean(samp,axis=0)
                    std_v[m]=np.std(samp,axis=0)
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    f=open(os.path.join(folder,'mcmc_summary.pckl'),'wb')
    pickle.dump([med_v,mean_v,std_v],f)
    f.close()

# plot 
plt.rcParams['image.cmap'] = 'binary'
num_rows=1
# posterior median
fig,axes = plt.subplots(nrows=num_rows,ncols=1+num_mdls,sharex=True,sharey=True,figsize=(16,4))
titles = ['Truth']+mdl_names
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        img=linv.misfit.truth
    else:
        img=med_v[i-1].reshape(linv.misfit.size)
    plt.imshow(img, origin='lower',extent=[0, 1, 0, 1])
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_med_comparepri.png',bbox_inches='tight')
# plt.show()

# posterior mean
fig,axes = plt.subplots(nrows=num_rows,ncols=1+num_mdls,sharex=True,sharey=True,figsize=(16,4))
titles = ['Truth']+mdl_names
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        img=linv.misfit.truth
    else:
        img=mean_v[i-1].reshape(linv.misfit.size)
    plt.imshow(img, origin='lower',extent=[0, 1, 0, 1])
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_mean_comparepri.png',bbox_inches='tight')
# plt.show()

# posterior std
fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls,sharex=True,sharey=True,figsize=(12,4))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    img=std_v[i].reshape(linv.misfit.size)
    plt.imshow(img, origin='lower',extent=[0, 1, 0, 1])
    ax.set_title(titles[i+1],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_std_comparepri.png',bbox_inches='tight')
# plt.show()