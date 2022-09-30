"""
Main function to run elliptic slice sampling for linear inverse problem
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
import timeit,time

# the inverse problem
from Linv import Linv

# MCMC
import sys
sys.path.append( "../" )
from sampler.ESS import ESS

np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('mdl_NO', nargs='?', type=int, default=0)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=5000)
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','qep'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    
    ## define the linear inverse problem ##
    prior_option = args.mdls[args.mdl_NO]
    fltnz = 2
    basis_opt = 'Fourier'
    KL_truc = 2000
    sigma = 1
    s = 1
    q = args.q
    if prior_option == 'gp': q=2 # this is not a tuning parameter for GP
    store_eig = True
    linv = Linv(prior_option=prior_option, fltnz=fltnz, basis_opt=basis_opt, KL_truc=KL_truc, sigma=sigma, s=s, q=q, store_eig=store_eig, seed=seed, normalize=True, weightedge=True)
    logLik = lambda u: -linv.misfit.cost(linv.prior.vec2fun(u))
    
    # initialization
    u=linv.prior.fun2vec(linv.misfit.obs.flatten())# + .1*linv.prior.sample()
    # u=linv.prior.sample()
    l=logLik(u)
    
    # run MCMC to generate samples
    print("Running the elliptic slice sampler (ESS) for %s prior model with random seed %d ..." % ({0:'Gaussian',1:'q-Exponential'}[args.mdl_NO], args.seed_NO))
    
    samp=[]; loglik=[]; times=[]
    # proc_info=[int(j/100*(args.num_samp+args.num_burnin)) for j in range(5,105,5)]
    prog=np.ceil((args.num_samp+args.num_burnin)*(.05+np.arange(0,1,.05)))
    beginning=timeit.default_timer()
    for i in range(args.num_samp+args.num_burnin):
        if i==args.num_burnin:
            # start the timer
            tic=timeit.default_timer()
            print('\nBurn-in completed; recording samples now...\n')
        # generate MCMC sample with given sampler
        u,l=ESS(u,l,linv.prior.sample,logLik)
        # display acceptance at intervals
        # if i in proc_info:
        #     print('\n %d%% iterations completed.' % (int(i/(args.num_samp+args.num_burnin)*100)))
        if i+1 in prog:
            print('{0:.0f}% has been completed.'.format(np.float(i+1)/(args.num_samp+args.num_burnin)*100))
        # save results
        loglik.append(l)
        if i>=args.num_burnin: samp.append(u)
        times.append(timeit.default_timer()-beginning)
    # stop timer
    toc=timeit.default_timer()
    time_=toc-tic
    print("\nAfter %g seconds, %d samples have been collected. \n" % (time_,args.num_samp))
    
    # store the results
    samp=np.stack(samp); loglik=np.stack(loglik);  times=np.stack(times)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'result')
    if not os.path.exists(savepath): os.makedirs(savepath)
    filename='linv_ESS_dim'+str(len(u))+'_'+prior_option+'_'+ctime+'.pckl'
    f=open(os.path.join(savepath,filename),'wb')
    pickle.dump([fltnz,basis_opt,KL_truc,sigma,s,args, samp,loglik,time_,times],f)
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