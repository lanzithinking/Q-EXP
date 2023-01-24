"""
Get relative error of mean for uncertainty field u in Advection-Diffusion inverse problem.
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Modified for STIP August 2021 @ ASU
"""

import os,pickle
import numpy as np
import dolfin as df
# import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mp

from advdiff import advdiff
import sys
sys.path.append( "../" )
from util.dolfin_gadget import *
from util.multivector import *


seed=2022
# define the inverse problem
meshsz = (61,61)
eldeg = 1
gamma = 1.; delta = 8.
L = 1000; store_eig = True
# observation_times = np.arange(1., 4.+.5*.1, .1)
rel_noise = .5
nref = 1
# adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
# adif.prior.V=adif.prior.Vh
# # adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# # get the true parameter (initial condition)
# ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
# true_param = df.interpolate(ic_expr, adif.prior.V).vector()


# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# store results
rem_m=np.zeros(num_mdls)
rem_s=np.zeros(num_mdls)
# obtain estimates
folder = './analysis_eldeg'+str(eldeg)
for m in range(num_mdls):
    prior_option=pri_mdls[m]
    q = 2 if prior_option=='gp' else 1
    # delta = 2. if  prior_option=='qep' else 10
    adif = advdiff(mesh=meshsz, eldeg=eldeg, prior_option=prior_option, gamma=gamma, delta=delta, q=q, L=L, store_eig=store_eig, rel_noise=rel_noise, nref=nref, seed=seed, STlik=True)
    adif.prior.V=adif.prior.Vh
    ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
    true_param = df.interpolate(ic_expr, adif.prior.V).vector()
    print('Processing '+pri_mdls[m]+' prior model...\n')
    fld_m = folder+'/'+pri_mdls[m]
    # preparation for estimates
    if os.path.exists(fld_m):
        hdf5_files=[f for f in os.listdir(fld_m) if f.endswith('.h5')]
        # pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
        num_samp=5000
        prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
        # calculate posterior estimates
        # found=False
        samp_f=df.Function(adif.prior.V,name="parameter")
        samp_mean=adif.prior.gen_vector(); samp_mean.zero()
        # samp_std=adif.prior.gen_vector(); samp_std.zero()
        errs=[]
        num_read=0
        for f_i in hdf5_files:
            try:
                f=df.HDF5File(adif.pde.mpi_comm,os.path.join(fld_m,f_i),"r")
                samp_mean.zero(); #samp_std.zero(); #num_read=0
                # err_=0
                for s in range(num_samp):
                    if s+1 in prog:
                        print('{0:.0f}% has been completed.'.format(np.float(s+1)/num_samp*100))
                    f.read(samp_f,'sample_{0}'.format(s))
                    u=samp_f.vector()
                    samp_mean.axpy(1./num_samp,u)
                    # samp_std.axpy(1./num_samp,u*u)
                    # err_+=(u-true_param).norm('l2')**2./num_samp
                # compute error
                errs.append((samp_mean-true_param).norm('l2')/true_param.norm('l2'))
                # errs.append(err_/true_param.norm('l2'))
                num_read+=1
                f.close()
                print(f_i+' has been read!')
                # f_read=f_i
            except:
                pass
        print('%d experiment(s) have been processed for %s prior model.' % (num_read, pri_mdls[m]))
        if num_read>0:
#             samp_mean=samp_mean/num_read; samp_std=samp_std/num_read
            errs = np.stack(errs)
            rem_m[m] = np.median(errs)
            rem_s[m] = errs.std()
            # get the best for plotting
            if not os.path.exists(os.path.join(fld_m,'mcmc_summary.h5')):
                print('Getting the summary to plot...\n')
                f_i=hdf5_files[np.argmin(errs)]
                f=df.HDF5File(adif.pde.mpi_comm,os.path.join(fld_m,f_i),"r")
                mean_v=adif.prior.gen_vector(); mean_v.zero()
                std_v=adif.prior.gen_vector(); std_v.zero()
                for s in range(num_samp):
                    if s+1 in prog:
                        print('{0:.0f}% has been completed.'.format(np.float(s+1)/num_samp*100))
                    f.read(samp_f,'sample_{0}'.format(s))
                    u=samp_f.vector()
                    mean_v.axpy(1./num_samp,u)
                    std_v.axpy(1./num_samp,u*u)
                f.close()
                std_v.axpy(-1,mean_v*mean_v)
                std_v.set_local(np.sqrt(std_v.get_local()))
                f=df.HDF5File(adif.pde.mpi_comm,os.path.join(fld_m,'mcmc_summary.h5'),"w")
                f.write(vec2fun(mean_v,adif.prior.V),'mean')
                f.write(vec2fun(std_v,adif.prior.V),'std')
                f.close()
                
# save
import pandas as pd
rem_m = pd.DataFrame(data=rem_m[None,:],columns=mdl_names[:num_mdls])
rem_s = pd.DataFrame(data=rem_s[None,:],columns=mdl_names[:num_mdls])
rem_m.to_csv(os.path.join(folder,'REM-mean.csv'),columns=mdl_names[:num_mdls])
rem_s.to_csv(os.path.join(folder,'REM-std.csv'),columns=mdl_names[:num_mdls])