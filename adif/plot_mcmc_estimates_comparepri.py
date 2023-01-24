"""
Plot estimates of uncertainty field u in Advection-Diffusion inverse problem.
Shiwei Lan @ U of Warwick, 2016
-------------------------------
Modified for STIP August 2021 @ ASU
"""

import os,pickle
import numpy as np
import dolfin as df
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
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
# adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()
# get the true parameter (initial condition)
ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
true_param = df.interpolate(ic_expr, adif.prior.V).vector()

# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './analysis_eldeg'+str(eldeg)
mean_v=MultiVector(adif.prior.gen_vector(),num_mdls)
std_v=MultiVector(adif.prior.gen_vector(),num_mdls)
for m in range(num_mdls):
    print('Processing '+pri_mdls[m]+' prior model...\n')
    fld_m = folder+'/'+pri_mdls[m]
    if os.path.exists(os.path.join(fld_m,'mcmc_summary.h5')):
        sumy_f=df.Function(adif.prior.V,name="mv")
        with df.HDF5File(adif.mpi_comm,os.path.join(fld_m,'mcmc_summary.h5'),"r") as f:
            f.read(sumy_f,'mean')
            mean_v[m].set_local(sumy_f.vector())
            f.read(sumy_f,'std')
            std_v[m].set_local(sumy_f.vector())
    else:
        prior_option=pri_mdls[m]
        q = 2 if prior_option=='gp' else 1
        # delta = 2. if  prior_option=='qep' else 10
        adif = advdiff(mesh=meshsz, eldeg=eldeg, prior_option=prior_option, gamma=gamma, delta=delta, q=q, L=L, store_eig=store_eig, rel_noise=rel_noise, nref=nref, seed=seed, STlik=True)
        adif.prior.V=adif.prior.Vh
        ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
        true_param = df.interpolate(ic_expr, adif.prior.V).vector()
        # preparation for estimates
        if os.path.exists(fld_m):
            hdf5_files=[f for f in os.listdir(fld_m) if f.endswith('.h5')]
            # pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
            num_samp=5000
            prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
            # calculate posterior estimates
            samp_f=df.Function(adif.prior.V,name="parameter")
            samp_mean=adif.prior.gen_vector(); samp_mean.zero()
            samp_std=adif.prior.gen_vector(); samp_std.zero()
            err_min=np.inf
            num_read=0
            for f_i in hdf5_files:
                try:
                    f=df.HDF5File(adif.pde.mpi_comm,os.path.join(fld_m,f_i),"r")
                    samp_mean.zero(); samp_std.zero(); #num_read=0
                    for s in range(num_samp):
                        if s+1 in prog:
                            print('{0:.0f}% has been completed.'.format(np.float(s+1)/num_samp*100))
                        f.read(samp_f,'sample_{0}'.format(s))
                        u=samp_f.vector()
                        samp_mean.axpy(1./num_samp,u)
                        samp_std.axpy(1./num_samp,u*u)
                    f.close()
                    # record the summary corresponding to the minimal error
                    err_=(samp_mean-true_param).norm('l2')/true_param.norm('l2')
                    print(f_i+' has been read!')
                    if err_<err_min:
                        mean_v[m].set_local(samp_mean)
                        std_v[m].set_local(np.sqrt((samp_std - samp_mean*samp_mean).get_local()))
                except:
                    pass
        # save
        sumy_f=df.Function(adif.prior.V,name="mv")
        with df.HDF5File(adif.mpi_comm,os.path.join(fld_m,'mcmc_summary.h5'),"w") as f:
            sumy_f.vector().set_local(mean_v[m])
            f.write(sumy_f,'mean')
            sumy_f.vector().set_local(std_v[m])
            f.write(sumy_f,'std')

# plot
# num_algs-=1
plt.rcParams['image.cmap'] = 'jet'
num_rows=1
# posterior mean
fig,axes = plt.subplots(nrows=num_rows,ncols=1+num_mdls,sharex=True,sharey=True,figsize=(16,4))
titles = ['Truth']+mdl_names
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    if i==0:
        # plot truth
        sub_figs[i]=df.plot(vec2fun(true_param,adif.prior.V))
        ax.set_title('Truth',fontsize=18)
        common_clim=sub_figs[i].get_clim()
    else:
        sub_figs[i]=df.plot(vec2fun(mean_v[i-1],adif.prior.V))
        sub_figs[i].set_clim(common_clim)
    ax.set_title(titles[i],fontsize=16)
    ax.set_aspect('auto')
    plt.axis([0, 1, 0, 1])
# set color bar
# from util.common_colorbar import common_colorbar
# fig=common_colorbar(fig,axes,sub_figs)
cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.02,axes.flat[0].get_position().y1-ax.get_position().y0])
norm = mp.colors.Normalize(vmin=common_clim[0], vmax=common_clim[1])
common_colbar = mp.colorbar.ColorbarBase(cax,norm=norm)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_mean_comparepri.png',bbox_inches='tight')
# plt.show()

# posterior std
fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls,sharex=True,sharey=True,figsize=(12,4))
sub_figs = [None]*len(axes.flat)
for i,ax in enumerate(axes.flat):
    plt.axes(ax)
    sub_figs[i]=df.plot(vec2fun(std_v[i],adif.prior.V))
    if i==0:
        common_clim=sub_figs[i].get_clim()
    else:
        sub_figs[i].set_clim(common_clim)
    ax.set_title(titles[i+1],fontsize=16)
    ax.set_aspect('auto')
    plt.axis([0, 1, 0, 1])
# set color bar
# from util.common_colorbar import common_colorbar
# fig=common_colorbar(fig,axes,sub_figs)
cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.02,axes.flat[0].get_position().y1-ax.get_position().y0])
norm = mp.colors.Normalize(vmin=common_clim[0], vmax=common_clim[1])
common_colbar = mp.colorbar.ColorbarBase(cax,norm=norm)
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# save plot
# fig.tight_layout()
plt.savefig(folder+'/mcmc_estimates_std_comparepri.png',bbox_inches='tight')
# plt.show()