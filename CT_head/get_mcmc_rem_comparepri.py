"""
Get relative error of MAP for uncertainty field u in linear inverse problem of Shepp-Logan head phantom.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from CT import CT

seed=2022
# define the inverse problem
CT_set='proj200_loc512'
data_set='head'
ker_opt = 'serexp'
basis_opt = 'Fourier'
KL_trunc = 5000
space = 'vec' #if ker_opt!='graphL' else 'fun'
sigma2 = 1e3
s = 1
q = 1
store_eig = True#(ker_opt!='graphL')
prior_params={'ker_opt':ker_opt,
              'basis_opt':basis_opt, # serexp param
              'KL_trunc':KL_trunc,
              'space':space,
              'sigma2':sigma2,
              's':s,
              'q':q,
              'store_eig':store_eig}
lik_params={'CT_set':CT_set,
            'data_set':data_set}
ct = CT(**lik_params, **prior_params, seed=seed, normalize=True, weightedge=True)
# truth = ct.misfit.truth

# models
pri_mdls=('GP','BSV','qEP')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# store results
rem_m=np.zeros(num_mdls)
rem_s=np.zeros(num_mdls)
# obtain estimates
folder = './analysis'
if not os.path.exists(os.path.join(folder,'mcmc_summary.pckl')):
    med_f=[[]]*num_mdls
    mean_f=[[]]*num_mdls
    std_f=[[]]*num_mdls
for m in range(num_mdls):
    # preparation
    prior_params['prior_option']={'GP':'gp','BSV':'bsv','qEP':'qep'}[pri_mdls[m]]
    prior_params['q']=2 if prior_params['prior_option']=='gp' else q
    prior_params['s']=2 if prior_params['prior_option']=='gp' else s
    ct = CT(**prior_params,**lik_params,seed=seed)
    truth = ct.misfit.truth
    print('Processing '+pri_mdls[m]+' prior model...\n')
    fld_m = folder+'/'+pri_mdls[m]
    # preparation for estimates
    if os.path.exists(fld_m):
        errs=[]; files_read=[]
        num_read=0
        npz_files=[f for f in os.listdir(fld_m) if f.endswith('.npz') and f.startswith('wpCN_')]
        for f_i in npz_files:
            try:
                f_read=np.load(os.path.join(fld_m,f_i))
                samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                if ct.prior.space=='vec': samp=ct.prior.vec2fun(samp.T).T
                samp_mean=np.mean(samp,axis=0).reshape(ct.misfit.size,order='F')
                # compute error
                errs.append(np.linalg.norm(samp_mean-truth)/np.linalg.norm(truth))
                files_read.append(f_i)
                num_read+=1
                print(f_i+' has been read!')
            except:
                pass
        print('%d experiment(s) have been processed for %s prior model.' % (num_read, pri_mdls[m]))
        if num_read>0:
            errs = np.stack(errs)
            rem_m[m] = np.median(errs)
            rem_s[m] = errs.std()
            # get the best for plotting
            if not os.path.exists(os.path.join(folder,'mcmc_summary.pckl')):
                f_i=files_read[np.argmin(errs)]
                f_read=np.load(os.path.join(fld_m,f_i))
                samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                if ct.prior.space=='vec': samp=ct.prior.vec2fun(samp.T).T
                med_f[m]=np.median(samp,axis=0).reshape(ct.misfit.size,order='F')
                mean_f[m]=np.mean(samp,axis=0).reshape(ct.misfit.size,order='F')
                std_f[m]=np.std(samp,axis=0).reshape(ct.misfit.size,order='F')
                print(f_i+' has been selected for plotting.')
if not os.path.exists(os.path.join(folder,'mcmc_summary.pckl')):
    f=open(os.path.join(folder,'mcmc_summary.pckl'),'wb')
    pickle.dump([truth,med_f,mean_f,std_f],f)
    f.close()

# save
import pandas as pd
rem_m = pd.DataFrame(data=rem_m[None,:],columns=mdl_names[:num_mdls])
rem_s = pd.DataFrame(data=rem_s[None,:],columns=mdl_names[:num_mdls])
rem_m.to_csv(os.path.join(folder,'REM-mean.csv'),columns=mdl_names[:num_mdls])
rem_s.to_csv(os.path.join(folder,'REM-std.csv'),columns=mdl_names[:num_mdls])