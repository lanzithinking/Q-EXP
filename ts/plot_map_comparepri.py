"""
Plot estimates of the process u in the time series problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
# from TS import TS
from misfit import misfit

seed=2022
# define the inverse problem
truth_option = 1
# cov_opt = 'matern'
# basis_opt = 'Fourier'
# KL_trunc = 100
# sigma2 = 1
# s = 1
# store_eig = True
# ts = TS(truth_option=truth_option, cov_opt=cov_opt, basis_opt=basis_opt, KL_trunc=KL_trunc, sigma2=sigma2, s=s, store_eig=store_eig, seed=seed)
msft = misfit(truth_option=truth_option, size=200)

# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './MAP'
if os.path.exists(os.path.join(folder,'ts_'+msft.truth_name+'_map_summary.pckl')):
    f=open(os.path.join(folder,'ts_'+msft.truth_name+'_map_summary.pckl'),'rb')
    truth,maps,funs,errs=pickle.load(f)
    f.close()
    print('mcmc_summary.pckl has been read!')
else:
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    maps=[]; funs=[]; errs=[]
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        # preparation for estimates
        for f_i in pckl_files:
            if '_'+msft.truth_name+'_' in f_i and '_'+pri_mdls[m]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    truth=f_read[0]
                    maps.append(f_read[1])
                    funs.append(np.pad(f_read[2],(0,1000-len(f_read[2])),mode='constant',constant_values=np.nan))
                    errs.append(np.pad(f_read[3],(0,1000-len(f_read[3])),mode='constant',constant_values=np.nan))
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    maps=np.stack(maps)
    funs=np.stack(funs)
    errs=np.stack(errs)
    # save
    f=open(os.path.join(folder,'ts_'+msft.truth_name+'_map_summary.pckl'),'wb')
    pickle.dump([truth,maps,funs,errs],f)
    f.close()

# plot 
# plt.rcParams['image.cmap'] = 'binary'
num_rows=1
# posterior median
fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls,sharex=True,sharey=True,figsize=(16,5))
titles = mdl_names
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    ax.plot(msft.times, msft.truth)
    ax.plot(msft.times, maps[i], linewidth=2, linestyle='--', color='red')
    ax.scatter(msft.times, msft.obs, color='orange')
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(os.path.join(folder,'ts_'+msft.truth_name+'_maps_comparepri.png'),bbox_inches='tight')
# plt.show()

# errors
N=400
fig,axes = plt.subplots(nrows=num_rows,ncols=2,sharex=True,sharey=False,figsize=(16,5))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    plt.plot(np.arange(1,N+1),{0:funs,1:errs}[i][:,:N].T)
    plt.yscale('log')
    ax.set_xlabel('iteration',fontsize=15)
    ax.set_ylabel({0:'negative posterior',1:'relative error'}[i],fontsize=15)
    ax.legend(labels=mdl_names,fontsize=14)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.25, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(os.path.join(folder,'ts_'+msft.truth_name+'_errs_comparepri.png'),bbox_inches='tight')
# plt.show()