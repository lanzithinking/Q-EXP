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
from TS import TS

seed=2022
# define the inverse problem
truth_option = 1
size = 200
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
ts = TS(truth_option=truth_option, cov_opt=cov_opt, basis_opt=basis_opt, KL_trunc=KL_trunc, space=space, sigma2=sigma2, s=s, store_eig=store_eig, seed=seed)
# truth = ts.misfit.truth

# models
pri_mdls=('GP','BSV','qEP')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './analysis/'+ts.misfit.truth_name
if os.path.exists(os.path.join(folder,'ts_'+ts.misfit.truth_name+'_mcmc_summary.pckl')):
    f=open(os.path.join(folder,'ts_'+ts.misfit.truth_name+'_mcmc_summary.pckl'),'rb')
    truth,med_f,mean_f,std_f=pickle.load(f)
    f.close()
    print('ts_'+ts.misfit.truth_name+'_mcmc_summary.pckl has been read!')
else:
    med_f=[[]]*num_mdls
    mean_f=[[]]*num_mdls
    std_f=[[]]*num_mdls
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        fld_m = folder+'/'+pri_mdls[m]
        # preparation for estimates
        prior_params['prior_option']={'GP':'gp','BSV':'bsv','qEP':'qep'}[pri_mdls[m]]
        prior_params['ker_opt']='serexp' if prior_params['prior_option']=='bsv' else ker_opt
        prior_params['q']=2 if prior_params['prior_option']=='gp' else q
        ts = TS(**prior_params,**lik_params,seed=seed)
        truth = ts.misfit.truth
        if os.path.exists(fld_m):
            pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
            for f_i in pckl_files:
                try:
                    f=open(os.path.join(fld_m,f_i),'rb')
                    f_read=pickle.load(f)
                    samp=f_read[-4]
                    if ts.prior.space=='vec': samp=ts.prior.vec2fun(samp.T).T
                    med_f[m]=np.median(samp,axis=0)
                    mean_f[m]=np.mean(samp,axis=0)
                    std_f[m]=np.std(samp,axis=0)
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    f=open(os.path.join(folder,'ts_'+ts.misfit.truth_name+'_mcmc_summary.pckl'),'wb')
    pickle.dump([truth,med_f,mean_f,std_f],f)
    f.close()

# plot 
# plt.rcParams['image.cmap'] = 'jet'
num_rows=1
# posterior mean/median with credible band
fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls,sharex=True,sharey=True,figsize=(16,5))
titles = mdl_names
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    ax.plot(ts.misfit.times, truth)
    ax.scatter(ts.misfit.times, ts.misfit.obs, color='orange')
    ax.plot(ts.misfit.times, mean_f[i], linewidth=2, linestyle='--', color='red')
    ax.fill_between(ts.misfit.times,mean_f[i]-1.96*std_f[i],mean_f[i]+1.96*std_f[i],color='b',alpha=.2)
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/ts_'+ts.misfit.truth_name+'_mcmc_estimates_comparepri.png',bbox_inches='tight')
# plt.show()
