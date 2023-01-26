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
# from Tesla import Tesla
from misfit import misfit

seed=2022
# define the inverse problem
start_date='2022-01-01'
end_date='2023-01-01'
# cov_opt = 'matern'
# basis_opt = 'Fourier'
# KL_trunc = 100
# sigma2 = 100
# s = 1
# store_eig = True
# ts = Tesla(start_date=start_date, end_date=end_date, cov_opt=cov_opt, basis_opt=basis_opt, KL_trunc=KL_trunc, sigma2=sigma2, s=s, store_eig=store_eig, seed=seed)
msft = misfit(start_date=start_date, end_date=end_date)

# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# store results
rmse=np.zeros(num_mdls)
loglik=np.zeros(num_mdls)
# obtain estimates
folder = './MAP'
if os.path.exists(os.path.join(folder,'Tesla_'+str(msft.size)+'days_map_rmse.pckl')):
    f=open(os.path.join(folder,'Tesla_'+str(msft.size)+'days_map_rmse.pckl'),'rb')
    dat,rmse,loglik=pickle.load(f)
    f.close()
    print('map_rmse.pckl has been read!')
else:
    pckl_files=[f for f in os.listdir(folder) if f.endswith('.pckl')]
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        # preparation for estimates
        for f_i in pckl_files:
            if '_'+str(msft.size)+'days_' in f_i and '_'+pri_mdls[m]+'_' in f_i:
                try:
                    f=open(os.path.join(folder,f_i),'rb')
                    f_read=pickle.load(f)
                    dat=f_read[0]
                    map=f_read[1]
                    rmse[m]=np.linalg.norm(map-dat)
                    loglik[m]=-msft.cost(map)
                    f.close()
                    print(f_i+' has been read!'); break
                except:
                    pass
    # save
    f=open(os.path.join(folder,'Tesla_'+str(msft.size)+'days_map_rmse.pckl'),'wb')
    pickle.dump([dat,rmse,loglik],f)
    f.close()

# save
import pandas as pd
sumry = pd.DataFrame(data=np.vstack((rmse,loglik)),columns=mdl_names[:num_mdls],index=['rmse','log-lik'])
sumry.to_csv(os.path.join(folder,'Tesla_'+str(msft.size)+'days-map_rmse.csv'),columns=mdl_names[:num_mdls])