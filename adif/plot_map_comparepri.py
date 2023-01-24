"""
Plot estimates of uncertainty field u in Advection-Diffusion inverse problem.
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


seed=2020
# define the inverse problem
meshsz = (61,61)
eldeg = 1
gamma = 1.; delta = 8.
rel_noise = .5
nref = 1
adif = advdiff(mesh=meshsz, eldeg=eldeg, gamma=gamma, delta=delta, rel_noise=rel_noise, nref=nref, seed=seed)
adif.prior.V=adif.prior.Vh
# get the true parameter (initial condition)
ic_expr = df.Expression('min(0.5,exp(-100*(pow(x[0]-0.35,2) +  pow(x[1]-0.7,2))))', element=adif.prior.V.ufl_element())
true_param = df.interpolate(ic_expr, adif.prior.V).vector()

# models
pri_mdls=('gp','bsv','qep')
mdl_names=['Gaussian','Besov','q-Exponential']
num_mdls=len(pri_mdls)

# plot
folder = './MAP'
plt.rcParams['image.cmap'] = 'jet'
num_rows=1
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
        # plot MAP
        try:
            f=df.XDMFFile(adif.mpi_comm, os.path.join(folder+'/'+pri_mdls[i-1],'MAP_'+pri_mdls[i-1]+'.xdmf'))
            # f=df.XDMFFile(adif.mpi_comm, os.path.join(os.getcwd(),'properties','MAP_0.xdmf' if i==1 else 'MAP.xdmf'))
            MAP=df.Function(adif.prior.V,name="MAP")
            f.read_checkpoint(MAP,'m',0)
            f.close()
            sub_figs[i]=df.plot(MAP)
            sub_figs[i].set_clim(common_clim)
            # fig.colorbar(sub_figs[i],ax=ax)
            ax.set_title(titles[i],fontsize=16)
        except Exception as e:
            print(e)
            pass
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
plt.savefig(folder+'/maps_comparepri.png',bbox_inches='tight')
# plt.show()
