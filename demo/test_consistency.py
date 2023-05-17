"""
This is to test the consistency of Q-EP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append( "../" )
from util.qep.qEP import qEP

# set random seed
np.random.seed(2022)

# set up
dim = 3
q = 1
prior_args={'ker_opt':'covf','cov_opt':'matern','l':.5,'q':q,'L':100}
store_eig=True
# define Q-EP process
x = np.linspace(0,1,dim)
qep = qEP(x=x, store_eig=store_eig, **prior_args)

# generate samples
n_samp = 10000
samp0_qep = qep.rnd(n=n_samp).T # sample by Q-EP definition

# get marginal
sub_dim = 2
x_sub = x[:sub_dim]
qep_sub = qEP(x=x_sub, store_eig=store_eig, **prior_args)

qep.ker.tomat()
qep_sub.ker.tomat()

samp1_qep = qep_sub.rnd(n=n_samp).T
# samp2plot = pd.DataFrame( np.vstack(( np.hstack((samp0_qep[:,:sub_dim],np.tile(0,(n_samp,1)))), np.hstack((samp1_qep,np.tile(1,(n_samp,1)))) )),
#                         columns=[str(j+1) for j in range(sub_dim)]+['source'])
# samp2plot.rename(columns={samp2plot.columns.values[-1]: 'source'},inplace=True)
samp2plot = pd.DataFrame( np.vstack(( np.hstack((samp0_qep,np.tile(0,(n_samp,1)))), np.hstack((samp1_qep,np.tile(np.nan,(n_samp,dim-sub_dim)),np.tile(1,(n_samp,1)))) )),
                        columns=[str(j+1) for j in range(dim)]+['source'])

sns.set(font_scale=1.02)
# plot
unique = samp2plot['source'].unique()
palette = dict(zip(unique, sns.color_palette(n_colors=len(unique),as_cmap=True)))
g=sns.PairGrid(data=samp2plot,vars=samp2plot.columns[:-1], hue='source', palette=palette)#, layout_pad=0.1)
g.map_upper(sns.scatterplot, s=1)#,style=samp2plot.loc[:,"source"])
# g.map_diag(sns.kdeplot)
# g.add_legend(legend_data={0:'marginal',1:'low-dimension'})
# plt.legend(labels=['marginal','low-dim'], frameon=False)
def diagkde(x, **kwargs):
    ax=plt.gca()
    sns.kdeplot(x=x.name, **kwargs)
    if x.name=='1':
        # plt.legend(loc='best',labels=['marginal','low-dim'], frameon=False)#, reverse=True)
        plt.legend(loc='best',labels=['marginal','full-dim'], frameon=False)
g.map_diag(diagkde, data=samp2plot, hue='source', legend=False)
g.map_lower(sns.kdeplot)
g.fig.tight_layout()
g.fig.subplots_adjust(top=.95)
g.fig.suptitle('Q-Exponential')
# plt.show()
for ax in g.axes.flatten():
    if ax:
        # rotate x axis labels
        if ax.get_xlabel()!='': ax.set_xlabel('$x_'+ax.get_xlabel()+'$')
        # rotate y axis labels
        if ax.get_ylabel()!='': ax.set_ylabel('$x_'+ax.get_ylabel()+'$', rotation = 0)
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right')
plt.savefig('./qep_consistency.png', bbox_inches='tight')