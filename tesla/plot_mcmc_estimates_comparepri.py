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
sigma2 = 100
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
# obtain estimates
folder = './analysis'
if os.path.exists(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days_mcmc_summary.pckl')):
    f=open(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days_mcmc_summary.pckl'),'rb')
    dat,med_f,mean_f,std_f=pickle.load(f)
    f.close()
    print('Tesla_'+str(ts.misfit.size)+'days_mcmc_summary.pckl has been read!')
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
        if prior_params['prior_option']=='gp':
            prior_params['sigma2'] = 100
        ts = Tesla(**prior_params,**lik_params,seed=seed)
        dat = ts.misfit.obs
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
    f=open(os.path.join(folder,'Tesla_'+str(ts.misfit.size)+'days_mcmc_summary.pckl'),'wb')
    pickle.dump([dat,med_f,mean_f,std_f],f)
    f.close()
    
import matplotlib.dates as mdates
# plot 
# plt.rcParams['image.cmap'] = 'jet'
num_rows=1
# posterior mean/median with credible band
fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls,sharex=True,sharey=True,figsize=(16,5))
titles = mdl_names
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.scatter(ts.misfit.times, ts.misfit.obs, color='orange',s=10)
    ax.plot(ts.misfit.times, mean_f[i], linewidth=2, linestyle='--', color='red')
    ax.fill_between(ts.misfit.times,mean_f[i]-1.96*std_f[i],mean_f[i]+1.96*std_f[i],color='b',alpha=.2)
    ax.set_title(titles[i],fontsize=18)
    plt.gcf().autofmt_xdate()
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/Tesla_'+str(ts.misfit.size)+'days_mcmc_estimates_comparepri.png',bbox_inches='tight')
# plt.show()
