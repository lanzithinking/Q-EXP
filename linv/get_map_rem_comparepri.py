"""
Get root mean squared error (RMSE) of MAP for the process u in the time series problem.
----------------------
Shiwei Lan @ ASU, 2022
"""

import os,pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp

# the inverse problem
from Linv import Linv
from misfit import misfit

seed=2022
# define the inverse problem
fltnz = 2
# basis_opt = 'Fourier'
# KL_trunc = 2000
# sigma2 = 1
# s = 1
# store_eig = True
# linv = Linv(fltnz=fltnz, basis_opt=basis_opt, KL_trunc=KL_trunc, sigma2=sigma2, s=s, store_eig=store_eig, seed=seed, normalize=True, weightedge=True)
msft = misfit(fltnz=fltnz)

# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# store results
rem=np.zeros(num_mdls)
loglik=np.zeros(num_mdls)
# obtain estimates
folder = './MAP'
if os.path.exists(os.path.join(folder,'map_rem.pckl')):
    f=open(os.path.join(folder,'map_rem.pckl'),'rb')
    truth,rem,loglik=pickle.load(f)
    f.close()
    print('map_rem.pckl has been read!')
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
                    truth=f_read[0]
                    map=f_read[1]
                    rem[m]=np.linalg.norm(map-truth)/np.linalg.norm(truth)
                    loglik[m]=-msft.cost(map)
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    f=open(os.path.join(folder,'map_rem.pckl'),'wb')
    pickle.dump([truth,rem,loglik],f)
    f.close()

# save
import pandas as pd
sumry = pd.DataFrame(data=np.vstack((rem,loglik)),columns=mdl_names[:num_mdls],index=['rem','log-lik'])
sumry.to_csv(os.path.join(folder,'map_rem.csv'),columns=mdl_names[:num_mdls])