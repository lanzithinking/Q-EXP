"""
Main function to run whitened preconditioned Crank-Nicolson sampling for the time series problem
----------------------
Shiwei Lan @ ASU, 2022
----------------------
created by Shuyi Li
"""

# modules
import os,argparse,pickle
import numpy as np
import timeit,time
from scipy import stats

# the inverse problem
from Tesla import Tesla

# MCMC
import sys
sys.path.append( "../" )
from sampler.wpCN import wpCN


np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('mdl_NO', nargs='?', type=int, default=0)
    parser.add_argument('ker_NO', nargs='?', type=int, default=0)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=5000)
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','bsv','qep'))
    parser.add_argument('kers', nargs='?', type=str, default=('covf','serexp'))
    parser.add_argument('alg_NO', nargs='?', type=int, default=0)
    parser.add_argument('step_sizes', nargs='?', type=float, default=(.01,))
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN',))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(args.seed_NO)
    # np.random.seed(seed)
    
    ## define the linear inverse problem ##
    prior_params={'prior_option':args.mdls[args.mdl_NO],
                  'ker_opt':'serexp' if args.mdls[args.mdl_NO]=='bsv' else args.kers[args.ker_NO],
                  'cov_opt':'matern',
                  'basis_opt':'Fourier', # serexp param
                  'KL_trunc':100,
                  'space':'fun',
                  'sigma2':100,# if args.mdls[args.mdl_NO]=='gp' else 1,
                  's':1,
                  'q':2 if args.mdls[args.mdl_NO]=='gp' else args.q,
                  'store_eig':True}
    lik_params={'start_date':'2022-01-01',
                'end_date':'2023-01-01'}
    ts = Tesla(**prior_params,**lik_params,seed=args.seed_NO)
    logLik = lambda u: -ts._get_misfit(u, MF_only=True, incldet=False)
    # transformation
    z = lambda x: 2*stats.norm.cdf(abs(x))-1
    lmd = lambda x,q=ts.prior.q: 2**(1/q)*np.sign(x)*stats.gamma.ppf(z(x),1/q)**(1/q)
    T = lambda x,q=ts.prior.q: ts.prior.C_act(lmd(x), 1/q)
    
    # initialization random noise epsilon
    u=np.random.randn({'vec':ts.prior.ker.L,'fun':ts.prior.ker.N}[ts.prior.space])
    l=logLik(T(u))
    
    # run MCMC to generate samples
    print("Running the whitened preconditioned Crank-Nicolson (wpCN) for %s prior model with %s kernel taking random seed %d ..." % ('Besov', args.kers[args.ker_NO], args.seed_NO))
    
    samp=[]; loglik=[]; times=[]
    accp=0; acpt=0
    prog=np.ceil((args.num_samp+args.num_burnin)*(.05+np.arange(0,1,.05)))
    beginning=timeit.default_timer()
    for i in range(args.num_samp+args.num_burnin):
        if i==args.num_burnin:
            # start the timer
            tic=timeit.default_timer()
            print('\nBurn-in completed; recording samples now...\n')
        # generate MCMC sample with given sampler
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
            samp.append(T(u))
            acpt+=acpt_ind
        times.append(timeit.default_timer()-beginning)
    # stop timer
    toc=timeit.default_timer()
    time_=toc-tic
    acpt/=args.num_samp
    print("\nAfter %g seconds, %d samples have been collected. \n" % (time_,args.num_samp))
    
    # store the results
    samp=np.stack(samp); loglik=np.stack(loglik);  times=np.stack(times)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'result')
    if not os.path.exists(savepath): os.makedirs(savepath)
    filename='Tesla_'+str(ts.misfit.size)+'days_wpCN_dim'+str(len(u))+'_'+prior_params['prior_option']+'_'+prior_params['ker_opt']+'_'+ctime+'.pckl'
    f=open(os.path.join(savepath,filename),'wb')
    pickle.dump([prior_params, lik_params, args, samp,loglik,time_,times],f)
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