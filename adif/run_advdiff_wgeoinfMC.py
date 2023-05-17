"""
Main function to run whitened (geometric) dimension-independent sampling for Advection-Diffusion inverse problem to generate posterior samples
Shiwei Lan @ ASU, 2023
"""

# modules
import os,argparse,pickle
import numpy as np
import dolfin as df

# the inverse problem
from advdiff import advdiff

# MCMC
import sys
sys.path.append( "../" )
from sampler.wht_geoinfMC_dolfin import wht_geoinfMC

np.set_printoptions(precision=3, suppress=True)
# seed=2020
# np.random.seed(seed)

def main(seed=2020):
    parser = argparse.ArgumentParser()
    parser.add_argument('alg_NO', nargs='?', type=int, default=0)
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('mdl_NO', nargs='?', type=int, default=0)
    parser.add_argument('ker_NO', nargs='?', type=int, default=1)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=2500)
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','bsv','qep'))
    parser.add_argument('kers', nargs='?', type=str, default=('covf','serexp','graphL'))
    parser.add_argument('step_sizes', nargs='?', type=float, default=[1e-6,1e-5,.004,.003,.003]) # [.001,.005,.005] simple likelihood model
    # parser.add_argument('step_sizes', nargs='?', type=float, default=[.0001,.0005,.0005,None,None]) # [.0002,.001,.001] simple likelihood model
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN','winfMALA','winfHMC','winfmMALA','winfmHMC'))
    args = parser.parse_args()
    
    # set random seed
    seed=args.seed_NO
    np.random.seed(seed)

    ## define Advection-Diffusion inverse problem ##
#     mesh = df.Mesh('ad_10k.xml')
    meshsz = (61,61)
    eldeg = 1
    prior_option = args.mdls[args.mdl_NO]
    gamma = 1.; delta = 8.
    q = args.q; L = 1000; store_eig = True
    if prior_option=='gp': q=2
    rel_noise = .5
    # rel_noise = .2
    # observation_times = np.arange(1., 4.+.5*.1, .1)
    nref = 1
    adif = advdiff(mesh=meshsz, eldeg=eldeg, prior_option=prior_option, gamma=gamma, delta=delta, q=q, L=L, store_eig=store_eig, rel_noise=rel_noise, nref=nref, seed=seed, STlik=True) # set STlik=False for simple likelihood
    adif.prior.V=adif.prior.Vh
    
    # initialization
    try:
        MAP_file=os.path.join(os.getcwd(),'properties/MAP.xdmf')
        unknown=df.Function(adif.prior.V, name='MAP')
        f=df.XDMFFile(adif.mpi_comm,MAP_file)
        f.read_checkpoint(unknown,'m',0)
        f.close()
        unknown = unknown.vector()
        # unknown=adif.prior.sample()
        unknown=adif.whtprior.u2v(unknown)
    except Exception as e:
        print(e)
        unknown=adif.whtprior.sample()
    h=1e-7; v=adif.whtprior.sample()
    l,g=adif.get_geom(unknown,geom_ord=[0,1],whitened=True)[:2]; hess=adif.get_geom(unknown,geom_ord=[1.5],whitened=True)[2]
    Hv=adif.whtprior.generate_vector()
    hess.mult(v,Hv)
    l1,g1=adif.get_geom(unknown+h*v,geom_ord=[0,1],whitened=True)[:2]
    print('error in gradient: %0.8f' %(abs((l1-l)/h-g.inner(v))/v.norm('l2')))
    print('error in Hessian: %0.8f' %((-(g1-g)/h-Hv).norm('l2')/v.norm('l2')))
    
    
    # run MCMC to generate samples
    print("Running %s sampler with step size %g for %d step(s) for %s prior model with %s kernel taking random seed %d ..." 
          % (args.algs[args.alg_NO],args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO], args.mdls[args.mdl_NO],args.kers[args.ker_NO], args.seed_NO))
    
    winfMC=wht_geoinfMC(unknown,adif,args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO],args.algs[args.alg_NO],transformation=adif.whtprior.v2u, MF_only=True, whitened=True)
    winfMC.sample(args.num_samp,args.num_burnin)
    
    # append PDE information including the count of solving
    filename_=os.path.join(winfMC.savepath,winfMC.filename+'.pckl')
    filename=os.path.join(winfMC.savepath,'AdvDiff_'+winfMC.filename+'.pckl') # change filename
    os.rename(filename_, filename)
    f=open(filename,'ab')
#     soln_count=[adif.soln_count,adif.pde.soln_count]
    soln_count=adif.pde.soln_count
    pickle.dump([meshsz,rel_noise,nref,soln_count,args],f)
    f.close()
#     # verify with load
#     f=open(filename,'rb')
#     mc_samp=pickle.load(f)
#     pde_info=pickle.load(f)
#     f.close
#     print(pde_cnt)

if __name__ == '__main__':
    main()
    # set random seed
    # seeds = [2020+i*10 for i in range(1,10)]
    # n_seed = len(seeds)
    # for i in range(n_seed):
    #     print("Running for seed %d ...\n"% (seeds[i]))
    #     np.random.seed(seeds[i])
    #     main(seed=seeds[i])
