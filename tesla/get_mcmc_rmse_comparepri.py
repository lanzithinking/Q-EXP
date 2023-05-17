"""
Get root mean squared error (RMSE) for the process u in the time series problem.
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
start_date='2022-01-01'
end_date='2023-01-01'
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
lik_params={'start_date':start_date,
            'end_date':end_date}
ts = Tesla(**prior_params,**lik_params,seed=seed)

# models
pri_mdls=('GP','BSV','qEP')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# store results
rmse_m=np.zeros(num_mdls)
rmse_s=np.zeros(num_mdls)
# obtain estimates
folder = './analysis'
if not os.path.exists(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days_mcmc_summary.pckl')):
    med_f=[[]]*num_mdls
    mean_f=[[]]*num_mdls
    std_f=[[]]*num_mdls
for m in range(num_mdls):
    # preparation
    prior_params['prior_option']={'GP':'gp','BSV':'bsv','qEP':'qep'}[pri_mdls[m]]
    # prior_params['ker_opt']='serexp' if prior_params['prior_option']=='bsv' else ker_opt
    prior_params['q']=2 if prior_params['prior_option']=='gp' else q
    prior_params['sigma2'] = 10 if prior_params['prior_option']=='gp' else sigma2
    ts = Tesla(**prior_params,**lik_params,seed=seed)
    dat = ts.misfit.obs
    print('Processing '+pri_mdls[m]+' prior model...\n')
    fld_m = folder+'/'+pri_mdls[m]
    # preparation for estimates
    if os.path.exists(fld_m):
        rmses=[]; files_read=[]
        num_read=0
        npz_files=[f for f in os.listdir(fld_m) if f.endswith('.npz')]
        for f_i in npz_files:
            try:
                f_read=np.load(os.path.join(fld_m,f_i))
                samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                if ts.prior.space=='vec': samp=ts.prior.vec2fun(samp.T).T
                samp_mean=np.mean(samp,axis=0)
                # compute error
                rmses.append(np.linalg.norm(samp_mean-dat.flatten()))#/np.linalg.norm(dat.flatten()))
                files_read.append(f_i)
                num_read+=1
                print(f_i+' has been read!')
            except:
                pass
        print('%d experiment(s) have been processed for %s prior model.' % (num_read, pri_mdls[m]))
        if num_read>0:
            rmses = np.stack(rmses)
            rmse_m[m] = np.median(rmses)
            rmse_s[m] = rmses.std()
            # get the best for plotting
            if not os.path.exists(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days_mcmc_summary.pckl')):
                f_i=files_read[np.argmin(rmses)]
                f_read=np.load(os.path.join(fld_m,f_i))
                samp=f_read['samp_u' if '_hp_' in f_i else 'samp']
                if ts.prior.space=='vec': samp=ts.prior.vec2fun(samp.T).T
                med_f[m]=np.median(samp,axis=0)
                mean_f[m]=np.mean(samp,axis=0)
                std_f[m]=np.std(samp,axis=0)
                print(f_i+' has been selected for plotting.')
if not os.path.exists(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days_mcmc_summary.pckl')):
    f=open(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days_mcmc_summary.pckl'),'wb')
    pickle.dump([dat,med_f,mean_f,std_f],f)
    f.close()

# save
import pandas as pd
rmse_m = pd.DataFrame(data=rmse_m[None,:],columns=mdl_names[:num_mdls])
rmse_s = pd.DataFrame(data=rmse_s[None,:],columns=mdl_names[:num_mdls])
rmse_m.to_csv(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days-RMSE-mean.csv'),columns=mdl_names[:num_mdls])
rmse_s.to_csv(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days-RMSE-std.csv'),columns=mdl_names[:num_mdls])