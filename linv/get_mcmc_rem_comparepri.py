"""
Get relative error of mean for uncertainty field u in linear inverse problem.
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
true_param = linv.misfit.truth.flatten()

# models
pri_mdls=('GP','BSV','qEP')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# store results
rem_m=np.zeros(num_mdls)
rem_s=np.zeros(num_mdls)
# obtain estimates
folder = './analysis'
for m in range(num_mdls):
    print('Processing '+pri_mdls[m]+' prior model...\n')
    fld_m = folder+'/'+pri_mdls[m]
    # preparation for estimates
    if os.path.exists(fld_m):
        errs=[]
        num_read=0
        pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
        for f_i in pckl_files:
            try:
                f=open(os.path.join(fld_m,f_i),'rb')
                f_read=pickle.load(f)
                samp=f_read[-4]
                if linv.prior.space=='vec': samp=linv.prior.vec2fun(samp.T).T
                samp_mean=np.mean(samp,axis=0)
                # compute error
                errs.append(np.linalg.norm(samp_mean-true_param)/np.linalg.norm(true_param))
                num_read+=1
                f.close()
                print(f_i+' has been read!')
            except:
                pass
        print('%d experiment(s) have been processed for %s prior model.' % (num_read, pri_mdls[m]))
        if num_read>0:
            errs = np.stack(errs)
            rem_m[m] = np.median(errs)
            rem_s[m] = errs.std()
# save
import pandas as pd
rem_m = pd.DataFrame(data=rem_m[None,:],columns=mdl_names[:num_mdls])
rem_s = pd.DataFrame(data=rem_s[None,:],columns=mdl_names[:num_mdls])
rem_m.to_csv(os.path.join(folder,'REM-mean.csv'),columns=mdl_names[:num_mdls])
rem_s.to_csv(os.path.join(folder,'REM-std.csv'),columns=mdl_names[:num_mdls])