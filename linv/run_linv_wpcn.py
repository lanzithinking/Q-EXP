"""
Main function to run elliptic slice sampling for linear inverse problem
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np

# the inverse problem
from Linv import Linv

# MCMC
import sys
sys.path.append( "../" )
from sampler.w_geoinfMC import geoinfMC


np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('mdl_NO', nargs='?', type=int, default=2)
    parser.add_argument('ker_NO', nargs='?', type=int, default=1)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=5000)
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','qep','bsv'))
    parser.add_argument('kers', nargs='?', type=str, default=('covf','serexp','graphL'))
    parser.add_argument('algNO', nargs='?', type=int, default=0)
    parser.add_argument('step_sizes', nargs='?', type=float, default=[.0001,.0001,.004,None,None])
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN','infMALA','infHMC','DRinfmMALA','DRinfmHMC'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    ## define the linear inverse problem ##
    prior_params={'prior_option':args.mdls[args.mdl_NO],
                  'ker_opt':'serexp' if args.mdls[args.mdl_NO]=='bsv' else args.kers[args.ker_NO],
                  'basis_opt':'Fourier', # serexp param
                  'KL_trunc':2000,
                  'space':'fun' if args.kers[args.ker_NO]=='graphL' else 'vec',
                  'sigma2':1,
                  's':1,
                  'q':2 if args.mdls[args.mdl_NO]=='gp' else args.q,
                  'store_eig':args.kers[args.ker_NO]!='graphL',
                  'normalize':True, # graphL param
                  'weightedge':True} # graphL param
    lik_params={'fltnz':2}
    linv = Linv(**prior_params,**lik_params,seed=seed)
    
    # initialization random noise epsilon
    unknown=np.random.randn(np.prod(linv.prior.sample().shape)) 
    #u=linv.prior.fun2vec(linv.misfit.obs.flatten())# + .1*linv.prior.sample()
    
    # run MCMC to generate samples
    print("Running the wpcn for %s prior model with random seed %d ..." % ('besov', args.seed_NO))
    
    
    inf_GMC=geoinfMC(unknown,linv,args.step_sizes[args.algNO],args.step_nums[args.algNO],args.algs[args.algNO])
  
    #inf_GMC.q=unknown[1:]; inf_GMC.dim=2
    geom_ord=[0]
    if any(s in args.algs[args.algNO] for s in ['MALA','HMC']): geom_ord.append(1)
    
    mc_fun=inf_GMC.sample
    mc_args=(args.num_samp,args.num_burnin)
    mc_fun(*mc_args)
    
    # append ODE information including the count of solving
    filename_=os.path.join(inf_GMC.savepath,inf_GMC.filename+'.pckl')
    filename=os.path.join(inf_GMC.savepath,'linv_'+inf_GMC.filename[:12]+'_'+prior_params['prior_option']+'_'+prior_params['ker_opt']+inf_GMC.filename[12:]+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
    #soln_count=emj.ode.soln_count
    pickle.dump([prior_params, lik_params, args],f)
    f.close()
    
    
    

if __name__ == '__main__':
    main()
    # n_seed = 10; i=0; n_success=0
    # while n_success < n_seed:
    #     seed_i=2022+i*10
    #     try:
    #         print("Running for seed %d ...\n"% (seed_i))
    #         main(seed=seed_i)
    #         n_success+=1
    #     except Exception as e:
    #         print(e)
    #         pass
    #     i+=1