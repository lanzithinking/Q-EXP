"""
Plot estimates of uncertainty field u in linear inverse problem of Shepp-Logan head phantom.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from CT import CT


# seed=2022
# # define the inverse problem
# CT_set='proj200_loc512'
# data_set='head'
# ker_opt = 'serexp'
# basis_opt = 'Fourier'
# KL_trunc = 5000
# space = 'vec' #if ker_opt!='graphL' else 'fun'
# sigma2 = 1e3
# s = 1
# q = 1
# store_eig = True#(ker_opt!='graphL')
# prior_params={'ker_opt':ker_opt,
#               'basis_opt':basis_opt, # serexp param
#               'KL_trunc':KL_trunc,
#               'space':space,
#               'sigma2':sigma2,
#               's':s,
#               'q':q,
#               'store_eig':store_eig}
# lik_params={'CT_set':CT_set,
#             'data_set':data_set}
# ct = CT(**lik_params, **prior_params, seed=seed, normalize=True, weightedge=True)
# # truth = ct.misfit.truth

# models
pri_mdls=('GP','BSV','qEP')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './analysis'
if os.path.exists(os.path.join(folder,'mcmc_summary.pckl')):
    f=open(os.path.join(folder,'mcmc_summary.pckl'),'rb')
    truth, med_f,mean_f,std_f=pickle.load(f)
    f.close()
    print('mcmc_summary.pckl has been read!')
else:
    med_f=[[]]*num_mdls
    mean_f=[[]]*num_mdls
    std_f=[[]]*num_mdls
    for m in range(num_mdls):
        # preparation
        prior_params['prior_option']={'GP':'gp','BSV':'bsv','qEP':'qep'}[pri_mdls[m]]
        prior_params['q']=2 if prior_params['prior_option']=='gp' else q
        prior_params['s']=2 if prior_params['prior_option']=='gp' else s
        ct = CT(**lik_params,**prior_params,seed=seed)
        truth = ct.misfit.truth
        print('Processing '+pri_mdls[m]+' prior model...\n')
        fld_m = folder+'/'+pri_mdls[m]
        # preparation for estimates
        if os.path.exists(fld_m):
            npz_files=[f for f in os.listdir(fld_m) if f.endswith('.npz') and f.startswith('wpCN_')]
            for f_i in npz_files:
                try:
                    f_read=np.load(os.path.join(fld_m,f_i))
                    samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                    if ct.prior.space=='vec': samp=ct.prior.vec2fun(samp.T).T
                    med_f[m]=np.median(samp,axis=0).reshape(ct.misfit.size,order='F')
                    mean_f[m]=np.mean(samp,axis=0).reshape(ct.misfit.size,order='F')
                    std_f[m]=np.std(samp,axis=0).reshape(ct.misfit.size,order='F')
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    f=open(os.path.join(folder,'mcmc_summary.pckl'),'wb')
    pickle.dump([truth,med_f,mean_f,std_f],f)
    f.close()

# plot 
plt.rcParams['image.cmap'] = 'gray'
num_rows=1
titles = ['Truth']+mdl_names

# posterior median
fig,axes = plt.subplots(nrows=num_rows,ncols=1+num_mdls,sharex=True,sharey=True,figsize=(16,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    img=truth if i==0 else med_f[i-1]
    img=img[88:400,88:400]
    plt.imshow(img,extent=[0, 1, 0, 1])
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_med_comparepri.png',bbox_inches='tight')
# plt.show()

# posterior mean
fig,axes = plt.subplots(nrows=num_rows,ncols=1+num_mdls,sharex=True,sharey=True,figsize=(16,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    img=truth if i==0 else mean_f[i-1]
    img=img[88:400,88:400]
    plt.imshow(img,extent=[0, 1, 0, 1])
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_mean_comparepri.png',bbox_inches='tight')
# plt.show()

# posterior std
fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls,sharex=True,sharey=True,figsize=(14,4))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    img=std_f[i]
    img=img[88:400,88:400]
    sub_figs[i]=plt.imshow(img,extent=[0, 1, 0, 1])
    ax.set_title(titles[i+1],fontsize=16)
    ax.set_aspect('auto')
    plt.colorbar()
# # set color bar
# from util.common_colorbar import common_colorbar
# fig=common_colorbar(fig,axes,sub_figs)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_std_comparepri.png',bbox_inches='tight')
# plt.show()