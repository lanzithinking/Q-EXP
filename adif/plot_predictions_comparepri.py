"""
Plot prediction of spatiotemporal observations in Advection-Diffusion inverse problem.
Shiwei Lan, October 2021 @ ASU
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
STATE=0; PARAMETER=1


seed=2020
# define the inverse problem
meshsz = (61,61)
eldeg = 1
gamma = 1.; delta = 8.
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
# adif.misfit.obs=np.array([dat.get_local() for dat in adif.misfit.d.data]).flatten()


# modify the problem for prediction
simulation_times = np.arange(0., 5.+.5*.1, .1)
observation_times = simulation_times
adif_pred = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise,
                    simulation_times=simulation_times, observation_times=observation_times, save_obs=False, seed=seed)
# obtain true trajectories
ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
true_param = df.interpolate(ic_expr, adif.prior.V).vector()
adif_pred.x[PARAMETER]=true_param; adif_pred.pde.solveFwd(adif_pred.x[STATE],adif_pred.x)
true_trj=adif_pred.misfit.observe(adif_pred.x)

# selective locations to aggregate difference in observations
cond = np.logical_or([abs(x-.25)<.01 or abs(x-.6)<.01 for x in adif.misfit.targets[:,0]],[abs(y-.4)<.01 or abs(y-.85)<.01 for y in adif.misfit.targets[:,1]]) # or abs(y-.85)<.01
slab_idx = np.where(cond)[0]

# load data
fld=os.getcwd()
try:
    f=open(os.path.join(fld,'AdvDiff_obs.pckl'),'rb')
    obs,noise_variance=pickle.load(f)
    f.close()
    print('Observation file has been read!')
except Exception as e:
    print(e)
    raise


# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)
# obtain estimates
folder = './analysis_eldeg'+str(eldeg)
pred_m=np.zeros((num_mdls,adif_pred.misfit.targets.shape[0],len(adif_pred.misfit.observation_times)))
pred_std=np.zeros((num_mdls,adif_pred.misfit.targets.shape[0],len(adif_pred.misfit.observation_times)))
err_m=np.zeros((num_mdls,len(adif_pred.misfit.observation_times)))
err_std=np.zeros((num_mdls,len(adif_pred.misfit.observation_times)))
if os.path.exists(os.path.join(folder,'predictions.npz')):
    loaded = np.load(file=os.path.join(folder,'predictions.npz'))
    pred_m, pred_std, err_m, err_std=list(map(loaded.get,['pred_m','pred_std','err_m','err_std']))
    print('Prediction data loaded!')
else:
    for m in range(num_mdls):
        print('Processing '+pri_mdls[m]+' prior model...\n')
        fld_m = folder+'/'+pri_mdls[m]
        # preparation for estimates
        if os.path.exists(fld_m):
            hdf5_files=[f for f in os.listdir(fld_m) if f.endswith('.h5')]
            # pckl_files=[f for f in os.listdir(fld_m) if f.endswith('.pckl')]
            num_samp=5000
            prog=np.ceil(num_samp*(.1+np.arange(0,1,.1)))
            # calculate posterior estimates
            found=False
            samp_f=df.Function(adif.prior.V,name="parameter")
            fwdout_mean=0; fwdout_std=0
            fwderr_mean=0; fwderr_std=0
    #         num_read=0
            for f_i in hdf5_files:
                try:
                    f=df.HDF5File(adif.pde.mpi_comm,os.path.join(fld_m,f_i),"r")
                    fwdout_mean=0; fwdout_std=0; fwderr_mean=0; fwderr_std=0; #num_read=0
                    for s in range(num_samp):
                        if s+1 in prog:
                            print('{0:.0f}% has been completed.'.format(np.float(s+1)/num_samp*100))
                        f.read(samp_f,'sample_{0}'.format(s))
                        u=samp_f.vector()
                        adif_pred.x[PARAMETER]=u; adif_pred.pde.solveFwd(adif_pred.x[STATE],adif_pred.x)
                        pred=adif_pred.misfit.observe(adif_pred.x)
                        fwdout_mean+=1./num_samp*pred
                        fwdout_std+=1./num_samp*pred**2
                        err=np.linalg.norm(pred[:,slab_idx]-true_trj[:,slab_idx],axis=1)
                        fwderr_mean+=1./num_samp*err
                        fwderr_std+=1./num_samp*err**2
#                         num_read+=1
                    f.close()
                    print(f_i+' has been read!')
                    f_read=f_i
                    found=True; break
                except:
                    pass
            if found:
    #             fwdout_mean=fwdout_mean/num_read; fwdout_std=fwdout_std/num_read
                pred_m[m]=fwdout_mean.T
                pred_std[m]=np.sqrt(fwdout_std - fwdout_mean**2).T
                err_m[m]=fwderr_mean.T
                err_std[m]=np.sqrt(fwderr_std - fwderr_mean**2).T
    # save
    np.savez_compressed(file=os.path.join(folder,'predictions'), pred_m=pred_m, pred_std=pred_std, err_m=err_m, err_std=err_std)

# plot prediction
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
# num_algs-=1
plt.rcParams['image.cmap'] = 'jet'

# all locations
# num_rows=1
# for k in range(adif.misfit.targets.shape[0]):
#     fig,axes = plt.subplots(nrows=num_rows,ncols=num_mdls,sharex=True,sharey=False,figsize=(11,8))
#     # k = 50 # select location to plot observations
#     for i,ax in enumerate(axes.flat):
#         ax.plot(adif_pred.misfit.observation_times,true_trj[:,k],color='red',linewidth=1.5)
#         ax.scatter(adif.misfit.observation_times,obs[:,k])
#         ax.plot(adif_pred.misfit.observation_times,pred_m[i][k],linestyle='--')
#         ax.fill_between(adif_pred.misfit.observation_times,pred_m[i][k]-1.96*pred_std[i][k],pred_m[i][k]+1.96*pred_std[i][k],color='b',alpha=.1)
#         ax.set_title(mdl_names[i])
#         ax.set_aspect('auto')
#         # plt.axis([0, 1, 0, 1])
#     plt.suptitle('x=%.3f, \t y=%.3f'% tuple(adif.misfit.targets[k]))
#     plt.subplots_adjust(wspace=0.1, hspace=0.2)
#     # save plot
#     # fig.tight_layout()
#     if not os.path.exists(folder+'/predictions'): os.makedirs(folder+'/predictions')
#     plt.savefig(folder+'/predictions/comparepri_k'+str(k)+'.png',bbox_inches='tight')
    # plt.show()

# fig,axes = plt.subplots(nrows=num_rows,ncols=2,sharex=False,sharey=False,figsize=(9,8))
num_rows=1
fig,axes = plt.subplots(nrows=num_rows,ncols=2,sharex=False,sharey=False,figsize=(14,6))
locs=[25,30]
lg=[]
for i,ax in enumerate(axes.flat):
    plt.axes(axes.flat[i])
    # if i==0:
    # #     df.plot(adif_pred.misfit.Vh.mesh())
    # #     ax.scatter(adif_pred.misfit.targets[slab_idx,0],adif_pred.misfit.targets[slab_idx,1])
    #     cred_cover = np.mean(np.logical_and(pred_m[0]-1.96*pred_std[0] < true_trj.T, true_trj.T < pred_m[0]+1.96*pred_std[0]),axis=0)
    #     ax.plot(adif_pred.misfit.observation_times,cred_cover,linestyle='--')
    #     cred_cover = np.mean(np.logical_and(pred_m[2]-1.96*pred_std[2] < true_trj.T, true_trj.T < pred_m[2]+1.96*pred_std[2]),axis=0)
    #     ax.plot(adif_pred.misfit.observation_times,cred_cover,linestyle='-.')
    #     ax.set_xlabel('t', fontsize=16)
    #     ax.set_title('truth covering rate of credible bands',fontsize=16)
    #     plt.legend([mdl_names[i] for i in [0,2]],frameon=False, fontsize=15)
    # # elif i==1:
    # #     ax.plot(adif_pred.misfit.observation_times,np.zeros(len(adif_pred.misfit.observation_times)),color='red',linewidth=1.5)
    # #     ax.plot(adif_pred.misfit.observation_times,err_m[0],linestyle='--')
    # #     ax.fill_between(adif_pred.misfit.observation_times,err_m[0]-1.96*err_std[0],err_m[0]+1.96*err_std[0],color='b',alpha=.1)
    # #     ax.plot(adif_pred.misfit.observation_times,err_m[2],linestyle='-.')
    # #     ax.fill_between(adif_pred.misfit.observation_times,err_m[2]-1.96*err_std[2],err_m[2]+1.96*err_std[2],color='y',alpha=.1)
    # else:
    k=locs[i]
    g=ax.plot(adif_pred.misfit.observation_times,true_trj[:,k],color='red',linewidth=1.5); lg.append(g[0])
    ax.scatter(adif.misfit.observation_times,obs[:,k])
    g=ax.plot(adif_pred.misfit.observation_times,pred_m[0][k],linestyle='--'); lg.append(g[0])
    ax.fill_between(adif_pred.misfit.observation_times,pred_m[0][k]-1.96*pred_std[0][k],pred_m[0][k]+1.96*pred_std[0][k],color='b',alpha=.1)
    g=ax.plot(adif_pred.misfit.observation_times,pred_m[2][k],linestyle='-.'); lg.append(g[0])
    ax.fill_between(adif_pred.misfit.observation_times,pred_m[2][k]-1.96*pred_std[2][k],pred_m[2][k]+1.96*pred_std[2][k],color='y',alpha=.2)
    ax.set_xlabel('t', fontsize=18)
    ax.set_title('forward prediction (x=%.3f, y=%.3f)'% tuple(adif.misfit.targets[k]),fontsize=18)
    plt.legend(lg, ['Truth']+[mdl_names[i] for i in [0,2]],frameon=False, fontsize=16)
    ax.set_aspect('auto')
    # plt.axis([0, 1, 0, 1])
plt.subplots_adjust(wspace=0.15, hspace=0.1)
# save plot
# fig.tight_layout()
# if not os.path.exists(folder+'/predictions'): os.makedirs(folder+'/predictions')
plt.savefig(folder+'/predictions_comparepri.png',bbox_inches='tight')
# plt.show()