"""
Get various metrics of MAP for torso CT.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle,sys
import numpy as np
from skimage.metrics import structural_similarity as ssim_f
sys.path.append( "../CT/" )
from haar_psi import haar_psi_numpy

# the inverse problem
# from CT import CT
from misfit import misfit

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
data_set='torso'
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
# store results
rem=np.zeros(num_mdls)
loglik=np.zeros(num_mdls)
psnr=np.zeros(num_mdls)
ssim=np.zeros(num_mdls)
haarpsi=np.zeros(num_mdls)
# obtain estimates
folder = './MAP'
if os.path.exists(os.path.join(folder,'map_met.pckl')):
    f=open(os.path.join(folder,'map_met.pckl'),'rb')
    truth,rem,loglik,psnr,ssim,haarpsi=pickle.load(f)
    f.close()
    print('map_met.pckl has been read!')
else:
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        # preparation for estimates
        for f_i in pckl_files:
            if '_'+pri_mdls[m]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    truth=f_read[1]
                    map_f=f_read[2]
                    rem[m]=np.linalg.norm(map_f-truth)/np.linalg.norm(truth)
                    loglik[m]=-msft.cost(map_f.flatten(order='F'))
                    psnr[m]=PSNR(map_f, truth)
                    ssim[m]=SSIM(map_f, truth)
                    map_ = ((map_f - np.min(map_f)) * (1/(np.max(map_f) - np.min(map_f)) * 255)) #.astype('uint8')
                    haarpsi[m]=haar_psi_numpy(map_,truth)[0]
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    f=open(os.path.join(folder,'map_met.pckl'),'wb')
    pickle.dump([truth,rem,loglik,psnr,ssim,haarpsi],f)
    f.close()

# save
import pandas as pd
sumry = pd.DataFrame(data=np.vstack((rem,loglik,psnr,ssim,haarpsi)),columns=mdl_names[:num_mdls],index=['rem','log-lik','psnr','ssim','haarpsi'])
sumry.to_csv(os.path.join(folder,'map_met.csv'),columns=mdl_names[:num_mdls])