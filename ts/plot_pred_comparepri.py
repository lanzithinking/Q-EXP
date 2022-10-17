"""
Plot predictions of the process u in the time series problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from TS import TS

seed=2022
# define the inverse problem
truth_option = 0
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
 # train index
tr_idx = np.hstack([np.arange(int(ts.misfit.size/2),dtype=int),int(ts.misfit.size/2)+np.arange(0,int(ts.misfit.size/8*3),2,dtype=int)])
te_idx = np.setdiff1d(np.arange(ts.misfit.size),tr_idx)

# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './MAP_hldout'
if os.path.exists(os.path.join(folder,'ts_'+ts.misfit.truth_name+'_prediction_summary.pckl')):
    f=open(os.path.join(folder,'ts_'+ts.misfit.truth_name+'_prediction_summary.pckl'),'rb')
    truth,maps,pred_m,pred_s=pickle.load(f)
    f.close()
    print('ts_'+ts.misfit.truth_name+'_prediction_summary.pckl has been read!')
else:
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    maps=[];pred_m=[];pred_s=[]
    for m in range(num_mdls):
        prior_params['prior_option']=pri_mdls[m]
        prior_params['ker_opt']='serexp' if prior_params['prior_option']=='bsv' else ker_opt
        prior_params['q']=2 if prior_params['prior_option']=='gp' else q
        ts = TS(**prior_params,**lik_params,seed=seed)
        truth = ts.misfit.truth
        C = ts.prior.ker.tomat()
        C11=C[np.ix_(te_idx,te_idx)]; C12=C[np.ix_(te_idx,tr_idx)]; C22=C[np.ix_(tr_idx,tr_idx)]
        pred_v=np.diag(C11-C12.dot(np.linalg.solve(C22,C12.T)))
        if prior_params['prior_option']=='qep': pred_v=pred_v*2**(2/ts.prior.q)*gamma(len(te_idx)/2+2/ts.prior.q)/(len(te_idx)*gamma(len(te_idx)/2))*len(te_idx)**(1-2/ts.prior.q)
        pred_s.append(np.sqrt(pred_v))
        print('Processing '+pri_mdls[m]+' prior model...\n')
        # preparation for estimates
        for f_i in pckl_files:
            if '_'+ts.misfit.truth_name+'_' in f_i and '_'+pri_mdls[m]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    truth=f_read[0]
                    maps.append(f_read[1])
                    pred_m.append(C12.dot(np.linalg.solve(C22,maps[m])))
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    maps=np.stack(maps)
    # save
    f=open(os.path.join(folder,'ts_'+ts.misfit.truth_name+'_prediction_summary.pckl'),'wb')
    pickle.dump([truth,maps,pred_m,pred_s],f)
    f.close()

# plot 
# plt.rcParams['image.cmap'] = 'binary'
num_rows=1
# posterior median
fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls-1,sharex=True,sharey=True,figsize=(12,5))
titles = mdl_names
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==1:i+=1
    ax.plot(ts.misfit.times, ts.misfit.truth)
    ax.plot(ts.misfit.times[tr_idx], maps[i], linewidth=1.5, linestyle='--', color='red', alpha=.6)
    ax.plot(ts.misfit.times[te_idx], pred_m[i], linewidth=2, linestyle='-.', color='red')
    ax.fill_between(ts.misfit.times[te_idx],pred_m[i]-1.96*pred_s[i],pred_m[i]+1.96*pred_s[i],color='red',alpha=.2)
    # ax.scatter(ts.misfit.times, ts.misfit.obs, color='orange')
    ax.plot([ts.misfit.times[tr_idx]]*2,[np.zeros(len(tr_idx))-(truth_option+1)*.5,.1*np.ones(len(tr_idx))-(truth_option+1)*.5], color='black', linewidth=.5)
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(os.path.join(folder,'ts_'+ts.misfit.truth_name+'_preds_comparepri.png'),bbox_inches='tight')
# plt.show()
