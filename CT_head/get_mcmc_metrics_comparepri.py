"""
Get various metrics of MCMC samples for uncertainty field u in linear inverse problem of Shepp-Logan head phantom.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle,sys
import numpy as np
from skimage.metrics import structural_similarity as ssim_f
sys.path.append( "../CT/" )
from haar_psi import haar_psi_numpy

# the inverse problem
from CT import CT

def PSNR(reco, gt):
    mse = np.mean((np.asarray(reco) - gt)**2)
    if mse == 0.:
        return float('inf')
    data_range = (np.max(gt) - np.min(gt))
    return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(reco, gt):
    data_range = (np.max(gt) - np.min(gt))
    return ssim_f(reco, gt, data_range=data_range)

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
rle_m=np.zeros(num_mdls); rle_s=np.zeros(num_mdls)
loglik_m=np.zeros(num_mdls); loglik_s=np.zeros(num_mdls)
psnr_m=np.zeros(num_mdls); psnr_s=np.zeros(num_mdls)
ssim_m=np.zeros(num_mdls); ssim_s=np.zeros(num_mdls)
haarpsi_m=np.zeros(num_mdls); haarpsi_s=np.zeros(num_mdls)
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
    prior_params['s']=1 if prior_params['prior_option']=='bsv' else s
    ct = CT(**prior_params,**lik_params,seed=seed)
    truth = ct.misfit.truth
    print('Processing '+pri_mdls[m]+' prior model...\n')
    fld_m = folder+'/'+pri_mdls[m]
    # preparation for estimates
    if os.path.exists(fld_m):
        errs=[]; loglik=[]; psnr=[]; ssim=[]; haarpsi=[]; files_read=[]
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
                loglik.append(np.mean(f_read['loglik']))
                psnr.append(PSNR(samp_mean, truth))
                ssim.append(SSIM(samp_mean, truth))
                samp_mean_ = ((samp_mean - np.min(samp_mean)) * (1/(np.max(samp_mean) - np.min(samp_mean)) * 255)) #.astype('uint8')
                haarpsi.append(haar_psi_numpy(samp_mean_,truth)[0])
                files_read.append(f_i)
                num_read+=1
                print(f_i+' has been read!')
            except:
                pass
        print('%d experiment(s) have been processed for %s prior model.' % (num_read, pri_mdls[m]))
        if num_read>0:
            errs = np.stack(errs)
            rle_m[m] = np.median(errs)
            rle_s[m] = errs.std()
            loglik_m[m] = np.median(loglik)
            loglik_s[m] = np.std(loglik)
            psnr_m[m] = np.median(psnr)
            psnr_s[m] = np.std(psnr)
            ssim_m[m] = np.median(ssim)
            ssim_s[m] = np.std(ssim)
            haarpsi_m[m] = np.median(haarpsi)
            haarpsi_s[m] = np.std(haarpsi)
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
means = pd.DataFrame(data=np.vstack((rle_m,loglik_m,psnr_m,ssim_m,haarpsi_m)),columns=mdl_names[:num_mdls],index=['rle','log-lik','psnr','ssim','haarpsi'])
stds = pd.DataFrame(data=np.vstack((rle_s,loglik_s,psnr_s,ssim_s,haarpsi_s)),columns=mdl_names[:num_mdls],index=['rle','log-lik','psnr','ssim','haarpsi'])
means.to_csv(os.path.join(folder,'MET-mean.csv'),columns=mdl_names[:num_mdls])
stds.to_csv(os.path.join(folder,'MET-std.csv'),columns=mdl_names[:num_mdls])