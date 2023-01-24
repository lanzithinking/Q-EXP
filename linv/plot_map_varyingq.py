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

# seed=2022
# # define the inverse problem
# fltnz = 2
# basis_opt = 'Fourier'
# KL_trunc = 2000
# sigma2 = 1
# s = 1
# store_eig = True
# linv = Linv(fltnz=fltnz, basis_opt=basis_opt, KL_trunc=KL_trunc, sigma2=sigma2, s=s, store_eig=store_eig, seed=seed, normalize=True, weightedge=True)

# models
qs=[0.5,1,1.5,2]
lbl_names=['Q-EP (q='+str(i)+')' for i in qs]
num_lbls=len(qs)
# obtain estimates
folder = './MAP'
if os.path.exists(os.path.join(folder,'map_varyingq.pckl')):
    f=open(os.path.join(folder,'map_varyingq.pckl'),'rb')
    truth,maps,funs,errs=pickle.load(f)
    f.close()
    print('map_varyingq.pckl has been read!')
else:
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    maps=[]; funs=[]; errs=[]
    for m in range(num_lbls):
        print('Processing q='+str(qs[m])+' ...\n')
        # preparation for estimates
        for f_i in pckl_files:
            if '_q'+str(qs[m])+'_' in f_i:
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
    f=open(os.path.join(folder,'map_varyingq.pckl'),'wb')
    pickle.dump([truth,maps,funs,errs],f)
    f.close()

# plot 
plt.rcParams['image.cmap'] = 'binary'
num_rows=1
titles=lbl_names[:-1]+['Gaussian (q=2)']
# MAP
fig,axes = plt.subplots(nrows=num_rows,ncols=num_lbls,sharex=True,sharey=True,figsize=(16,4))
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    img=maps[i]
    plt.imshow(img, origin='lower',extent=[0, 1, 0, 1])
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/maps_varyingq.png',bbox_inches='tight')
# plt.show()

