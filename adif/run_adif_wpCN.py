"""
Main function to run whitened preconditioned Crank-Nicolson sampling for Advection-Diffusion inverse problem
Shiwei Lan @ ASU, 2020
"""

# modules
import os,argparse,pickle
import numpy as np
import dolfin as df
from scipy import stats
import timeit,time

# the inverse problem
from advdiff import advdiff

# MCMC
import sys
sys.path.append( "../" )
from sampler.wpCN_dolfin import wpCN

np.set_printoptions(precision=3, suppress=True)
# seed=2020
# np.random.seed(seed)

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('mdl_NO', nargs='?', type=int, default=1)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=5000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=2500)
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','bsv','qep'))
    parser.add_argument('alg_NO', nargs='?', type=int, default=0)
    parser.add_argument('step_sizes', nargs='?', type=float, default=(.01,))
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN',))
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
    logLik = lambda u: -adif._get_misfit(u)
    # rnd_pri = lambda: np.random.randn(adif.prior.N) if prior_option=='bsv' else adif.prior.sample()
    # transformation
    z = lambda x: 2*stats.norm.cdf(abs(x))-1
    lmd = lambda x,q=q: 2**(1/q)*np.sign(x)*stats.gamma.ppf(z(x),1/q)**(1/q)
    T = lambda x,q=q: adif.prior.C_act(lmd(x if isinstance(x,np.ndarray) else x.get_local()), 1/q)
    
    # initialization
    # if prior_option=='bsv':
    #     u=rnd_pri()
    # else:
    #     MAP_file=os.path.join(os.getcwd(),'./MAP/'+prior_option+'/MAP_'+prior_option+'.xdmf')
    #     if os.path.isfile(MAP_file):
    #         u_f=df.Function(adif.prior.V, name='MAP')
    #         f=df.XDMFFile(adif.mpi_comm,MAP_file)
    #         f.read_checkpoint(u_f,'m',0)
    #         f.close()
    #         u = u_f.vector()
    #     else:
    #         u=adif.get_MAP(SAVE=True)
    u=adif.prior.gen_vector(np.random.randn(adif.prior.dim))
    l=logLik(T(u) if prior_option=='bsv' else u)
    
    # run MCMC to generate samples
    print("Running the whitened preconditioned Crank-Nicolson (wpCN) for %s prior model with random seed %d ..." % ({0:'Gaussian',1:'Besov',2:'q-Exponential'}[args.mdl_NO], args.seed_NO))
    
    # allocate space to store results
    samp_fname='_samp_'+prior_option+'_dim'+str(adif.prior.V.dim())+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")
    samp_fpath=os.path.join(os.getcwd(),'result')
    if not os.path.exists(samp_fpath): os.makedirs(samp_fpath)
#         self.samp=df.File(os.path.join(samp_fpath,samp_fname+".xdmf"))
    samp=df.HDF5File(adif.pde.mpi_comm,os.path.join(samp_fpath,samp_fname+".h5"),"w")
    loglik=[]; times=[]
    accp=0; acpt=0
    prog=np.ceil((args.num_samp+args.num_burnin)*(.05+np.arange(0,1,.05)))
    beginning=timeit.default_timer()
    for i in range(args.num_samp+args.num_burnin):
        if i==args.num_burnin:
            # start the timer
            tic=timeit.default_timer()
            print('\nBurn-in completed; recording samples now...\n')
        # generate MCMC sample with given sampler
        # u,l=ESS(u,l,rnd_pri,lambda u:logLik(T(u)) if prior_option=='bsv' else logLik(u))
        u,l,acpt_ind=wpCN(u,l,logLik,T,args.step_sizes[args.alg_NO])
        # display acceptance at intervals
        if i+1 in prog:
            print('{0:.0f}% has been completed.'.format(np.float(i+1)/(args.num_samp+args.num_burnin)*100))
        # online acceptance rate
        accp+=acpt_ind
        if (i+1)%100==0:
            print('Acceptance at %d iterations: %0.2f' % (i+1,accp/100))
            accp=0.0
        # save results
        loglik.append(l)
        if i>=args.num_burnin:
            u_f=df.Function(adif.prior.V)
            u_f.vector().set_local(T(u) if prior_option=='bsv' else u)
            u_f.vector().apply('insert')
            samp.write(u_f,'sample_{0}'.format(i-args.num_burnin))
            acpt+=acpt_ind
        times.append(timeit.default_timer()-beginning)
    samp.close()
    # stop timer
    toc=timeit.default_timer()
    time_=toc-tic
    acpt/=args.num_samp
    print("\nAfter %g seconds, %d samples have been collected. \n" % (time_,args.num_samp))
    
    # store the results
    loglik=np.stack(loglik);  times=np.stack(times)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'result')
    if not os.path.exists(savepath): os.makedirs(savepath)
    filename='adif_wpCN_dim'+str(adif.prior.V.dim())+'_'+prior_option+'_'+ctime+'.pckl'
    f=open(os.path.join(savepath,filename),'wb')
    # append PDE information including the count of solving
#     soln_count=[adif.soln_count,adif.pde.soln_count]
    soln_count=adif.pde.soln_count
    pickle.dump([meshsz,rel_noise,nref,soln_count,args,loglik,time_,times],f)
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
    # seeds = [2022+i*10 for i in range(1,10)]
    # n_seed = len(seeds)
    # for i in range(n_seed):
    #     print("Running for seed %d ...\n"% (seeds[i]))
    #     main(seed=seeds[i])
