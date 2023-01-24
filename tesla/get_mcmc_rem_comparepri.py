"""
Get relative error of mean for the process u in the time series problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from Tesla import Tesla

seed=2022
# define the inverse problem
truth_option = 2
size = 252 if truth_option==0 else 100 if truth_option==1 else 60
ker_opt = 'covf'
cov_opt = 'matern'
basis_opt = 'Fourier'
KL_trunc = 100
space = 'fun'
sigma2 = 1
s = 1
q = 1
store_eig = True
prior_params={'ker_opt':ker_opt,
              'cov_opt':cov_opt,
              'basis_opt':basis_opt, # serexp param
              'KL_trunc':KL_trunc,
              'space':space,
              'sigma2':sigma2,
              's':s,
              'q':q,
              'store_eig':store_eig}
lik_params={'truth_option':truth_option,
                'size':size}
ts = Tesla(**prior_params,**lik_params,seed=seed)
# truth = ts.misfit.truth

# models
pri_mdls=('GP','BSV','qEP')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# store results
rem_m=np.zeros(num_mdls)
rem_s=np.zeros(num_mdls)
# obtain estimates
folder = './analysis/'+'nl0.05_50' #ts.misfit.truth_name
if not os.path.exists(os.path.join(folder,'Tesla_'+ts.misfit.truth_name+'_mcmc_summary.pckl')):
    med_f=[[]]*num_mdls
    mean_f=[[]]*num_mdls
    std_f=[[]]*num_mdls
for m in range(num_mdls):
    # preparation
    prior_params['prior_option']={'GP':'gp','BSV':'bsv','qEP':'qep'}[pri_mdls[m]]
    prior_params['ker_opt']='serexp' if prior_params['prior_option']=='bsv' else ker_opt
    prior_params['q']=2 if prior_params['prior_option']=='gp' else q
    if prior_params['prior_option']=='gp':
        prior_params['sigma2'] = 100
    ts = Tesla(**prior_params,**lik_params,seed=seed)
    truth = ts.misfit.truth
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
                if ts.prior.space=='vec': samp=ts.prior.vec2fun(samp.T).T
                samp_mean=np.mean(samp,axis=0)
                # compute error
                errs.append(np.linalg.norm(samp_mean-truth.flatten())/np.linalg.norm(truth.flatten()))
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
            # get the best for plotting
            if not os.path.exists(os.path.join(folder,'Covid_'+ts.misfit.truth_name+'_mcmc_summary.pckl')):
                f_i=pckl_files[np.argmin(errs)]
                f=open(os.path.join(fld_m,f_i),'rb')
                f_read=pickle.load(f)
                samp=f_read[-4]
                if ts.prior.space=='vec': samp=ts.prior.vec2fun(samp.T).T
                med_f[m]=np.median(samp,axis=0)
                mean_f[m]=np.mean(samp,axis=0)
                std_f[m]=np.std(samp,axis=0)
                f.close()
if not os.path.exists(os.path.join(folder,'Tesla_'+ts.misfit.truth_name+'_mcmc_summary.pckl')):
    f=open(os.path.join(folder,'Tesla_'+ts.misfit.truth_name+'_mcmc_summary.pckl'),'wb')
    pickle.dump([truth,med_f,mean_f,std_f],f)
    f.close()

# save
import pandas as pd
rem_m = pd.DataFrame(data=rem_m[None,:],columns=mdl_names[:num_mdls])
rem_s = pd.DataFrame(data=rem_s[None,:],columns=mdl_names[:num_mdls])
rem_m.to_csv(os.path.join(folder,'Tesla_'+ts.misfit.truth_name+'-REM-mean.csv'),columns=mdl_names[:num_mdls])
rem_s.to_csv(os.path.join(folder,'Tesla_'+ts.misfit.truth_name+'-REM-std.csv'),columns=mdl_names[:num_mdls])