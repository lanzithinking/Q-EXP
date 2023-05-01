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
# from CT import CT
from misfit import misfit

seed=2022
# define the inverse problem
CT_set='proj200_loc512'
data_set='head'
# basis_opt = 'Fourier'
# KL_trunc = 5000
# sigma2 = 1e3
# s = 1
# store_eig = True
# ct = CT(CT_set=CT_set, data_set=data_set, basis_opt=basis_opt, KL_trunc=KL_trunc, sigma2=sigma2, s=s, store_eig=store_eig, seed=seed, normalize=True, weightedge=True)
msft = misfit(CT_set=CT_set, data_set=data_set)

# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './MAP'
if os.path.exists(os.path.join(folder,'map_summary.pckl')):
    f=open(os.path.join(folder,'map_summary.pckl'),'rb')
    truth,maps,funs,errs=pickle.load(f)
    f.close()
    print('map_summary.pckl has been read!')
else:
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    maps=[]; funs=[]; errs=[]
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        # preparation for estimates
        for f_i in pckl_files:
            if '_'+pri_mdls[m]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    truth=f_read[1]
                    maps.append(f_read[2])
                    funs.append(np.pad(f_read[3],(0,1000-len(f_read[3])),mode='constant',constant_values=np.nan))
                    errs.append(np.pad(f_read[4],(0,1000-len(f_read[4])),mode='constant',constant_values=np.nan))
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    maps=np.stack(maps)
    funs=np.stack(funs)
    errs=np.stack(errs)
    # save
    f=open(os.path.join(folder,'map_summary.pckl'),'wb')
    pickle.dump([truth,maps,funs,errs],f)
    f.close()

# # plot 
# plt.rcParams['image.cmap'] = 'gray'
# num_rows=1
# # posterior median
# fig,axes = plt.subplots(nrows=num_rows,ncols=1+num_mdls,sharex=True,sharey=True,figsize=(16,4))
# titles = ['Truth']+mdl_names
# for i,ax in enumerate(axes.flat):
#     plt.axes(ax)
#     img=truth if i==0 else maps[i-1]
#     plt.imshow(img, origin='lower',extent=[0, 1, 0, 1])
#     ax.set_title(titles[i],fontsize=16)
#     ax.set_aspect('auto')
# plt.subplots_adjust(wspace=0.1, hspace=0.2)
# # save plot
# # fig.tight_layout()
# plt.savefig(folder+'/maps_comparepri.png',bbox_inches='tight')
# # plt.show()

A, phi, s, x_true, b=msft._gen_shepp_logan()
# plot 
plt.rcParams['image.cmap'] = 'gray'
num_rows=1
# map
fig,axes = plt.subplots(nrows=num_rows,ncols=2+num_mdls,sharex=False,sharey=True,figsize=(20,4))
titles = ['Truth','Observation']+mdl_names
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    img=truth if i==0 else msft.obs.reshape((phi.size,s.size),order='F').T if i==1 else maps[i-2]
    plt.imshow(img, origin='lower',extent=[0, 1, 0, 1])
    if i==1: ax.set_xticklabels(['$\pi$','$-4\pi/5$','$-3\pi/5$','$-2\pi/5$','$-\pi/5$','$0$'])
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/maps_comparepri.png',bbox_inches='tight')
# plt.show()

# errors
N=400
fig,axes = plt.subplots(nrows=num_rows,ncols=2,sharex=True,sharey=False,figsize=(12,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    plt.plot(np.arange(1,N+1),{0:funs,1:errs}[i][:,:N].T)
    plt.yscale('log')
    ax.set_xlabel('iteration',fontsize=15)
    ax.set_ylabel({0:'negative posterior',1:'relative error'}[i],fontsize=15)
    ax.legend(labels=mdl_names,fontsize=14)
    ax.set_aspect('auto')
plt.yticks([.2,.3,.4,.6], [ f"{x:.1f}" for x in [.2,.3,.4,.6] ])
plt.subplots_adjust(wspace=0.2, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/errs_comparepri.png',bbox_inches='tight')
# plt.show()