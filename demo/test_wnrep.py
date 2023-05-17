"""
This is to test the white noise representation of Q-EP
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
dim = 4
q = 1
prior_args={'ker_opt':'covf','cov_opt':'matern','l':.5,'q':q,'L':100}
store_eig=True
# define Q-EP process
x = np.linspace(0,1,dim)
qep = qEP(x=x, store_eig=store_eig, **prior_args)

# white noise representation
nmlz = lambda z,q=1: z/np.linalg.norm(z,axis=1)[:,None]**q
Lmd = lambda z,q=qep.q: qep.ker.act(nmlz(z.reshape((-1,qep.ker.N),order='F'),1-2/q),alpha=0.5,transp=True)

# generate samples
n_samp = 10000
samp_qep = qep.rnd(n=n_samp).T # sample by Q-EP definition
samp_wnr = Lmd(np.random.randn(n_samp,dim)) # sample by white noise representation
samp2plot = pd.DataFrame( np.vstack(( np.hstack((samp_qep,np.tile(0,(n_samp,1)))), np.hstack((samp_wnr,np.tile(1,(n_samp,1)))) )),
                        columns=[str(j+1) for j in range(dim)]+['source'])
# samp2plot.rename(columns={samp2plot.columns.values[-1]: 'source'},inplace=True)

sns.set(font_scale=1.02)
# plot
unique = samp2plot['source'].unique()
palette = dict(zip(unique, sns.color_palette(n_colors=len(unique),as_cmap=True)))
g=sns.PairGrid(data=samp2plot,vars=samp2plot.columns[:-1], hue='source', palette=palette)
g.map_upper(sns.scatterplot, s=1)#,style=samp2plot.loc[:,"source"])
# g.map_diag(sns.kdeplot)
def diagkde(x, **kwargs):
    ax=plt.gca()
    sns.kdeplot(x=x.name, **kwargs)
    if x.name=='1':
        plt.legend(loc='best',labels=['stoch-rep','wn-rep'], frameon=False)
g.map_diag(diagkde, data=samp2plot, hue='source', legend=False)
g.map_lower(sns.kdeplot)
# g.add_legend(title='method')#,legend_data={0:'Q-EP',1:'wn-rep'})
# plt.legend(labels=['Q-EP','wn-rep'])
for ax in g.axes.flatten():
    if ax:
        # rotate x axis labels
        if ax.get_xlabel()!='': ax.set_xlabel('$x_'+ax.get_xlabel()+'$')
        # rotate y axis labels
        if ax.get_ylabel()!='': ax.set_ylabel('$x_'+ax.get_ylabel()+'$', rotation = 0)
        # set y labels alignment
        ax.yaxis.get_label().set_horizontalalignment('right')
# plt.show()
plt.savefig('./qep_wnrep.png', bbox_inches='tight')