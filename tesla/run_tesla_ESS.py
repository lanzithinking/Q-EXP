"""
Main function to run elliptic slice sampling for the time series problem
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
from scipy import stats
import timeit,time

# the inverse problem
from Tesla import Tesla

# MCMC
import sys
sys.path.append( "../" )
from sampler.ESS import ESS

np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    parser = argparse.ArgumentParser()
    # parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('mdl_NO', nargs='?', type=int, default=0)
    parser.add_argument('ker_NO', nargs='?', type=int, default=0)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=5000)
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','bsv','qep'))
    parser.add_argument('kers', nargs='?', type=str, default=('covf','serexp'))
    args = parser.parse_args()
    
    # set random seed
    # np.random.seed(args.seed_NO)
    np.random.seed(seed)
    opt=2
    size = 252 if opt==0 else 100 if opt==1 else 60
    ## define the linear inverse problem ##
    prior_params={'prior_option':args.mdls[args.mdl_NO],
                  'ker_opt':'serexp' if args.mdls[args.mdl_NO]=='bsv' else args.kers[args.ker_NO],
                  'cov_opt':'matern',
                  'basis_opt':'Fourier', # serexp param
                  'KL_trunc':100,
                  'space':'fun',
                  'sigma2':1,
                  's':1,
                  'q':2 if args.mdls[args.mdl_NO]=='gp' else args.q,
                  'store_eig':True}
    lik_params={'truth_option':opt,
                'size':size}
    if args.mdls[args.mdl_NO]=='gp':
        prior_params['sigma2'] = 100
    # ts = TS(**prior_params,**lik_params,seed=args.seed_NO)
    ts = Tesla(**prior_params,**lik_params,seed=seed)
    logLik = lambda u: -ts._get_misfit(u, MF_only=True, incldet=False)
    rnd_pri = lambda: np.random.randn({'vec':ts.prior.ker.L,'fun':ts.prior.ker.N}[ts.prior.space]) if prior_params['prior_option']=='bsv' else ts.prior.sample()
    # transformation
    z = lambda x: 2*stats.norm.cdf(abs(x))-1
    lmd = lambda x,q=prior_params['q']: 2**(1/q)*np.sign(x)*stats.gamma.ppf(z(x),1/q)**(1/q)
    T = lambda x,q=prior_params['q']: ts.prior.C_act(lmd(x), 1/q)
    
    # initialization
    if prior_params['prior_option']=='bsv':
        u=rnd_pri()
        l=logLik(T(u))
    else:
        u=ts.misfit.obs.flatten()
        if ts.prior.space=='vec': u=ts.prior.fun2vec(u)
        # u+=.1*ts.prior.sample()
        l=logLik(u)
    
    # run MCMC to generate samples
    # print("Running the elliptic slice sampler (ESS) for %s prior model with %s kernel taking random seed %d ..." % ({0:'Gaussian',1:'Besov',2:'q-Exponential'}[args.mdl_NO], args.kers[args.ker_NO], args.seed_NO))
    print("Running the elliptic slice sampler (ESS) for %s prior model with %s kernel taking random seed %d ..." % ({0:'Gaussian',1:'Besov',2:'q-Exponential'}[args.mdl_NO], args.kers[args.ker_NO], seed))
    
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
        u,l=ESS(u,l,rnd_pri,lambda u:logLik(T(u)) if prior_params['prior_option']=='bsv' else logLik(u))
        # display acceptance at intervals
        # if i in proc_info:
        #     print('\n %d%% iterations completed.' % (int(i/(args.num_samp+args.num_burnin)*100)))
        if i+1 in prog:
            print('{0:.0f}% has been completed.'.format(np.float(i+1)/(args.num_samp+args.num_burnin)*100))
        # save results
        loglik.append(l)
        if i>=args.num_burnin: samp.append(T(u) if prior_params['prior_option']=='bsv' else u)
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
    filename='Tesla_'+ts.misfit.truth_name+'_ESS_dim'+str(len(u))+'_'+prior_params['prior_option']+'_'+prior_params['ker_opt']+'_'+ctime+'.pckl'
    f=open(os.path.join(savepath,filename),'wb')
    pickle.dump([prior_params, lik_params, args, samp,loglik,time_,times],f)
    f.close()

if __name__ == '__main__':
    # main()
    n_seed = 9; i=1; n_success=0
    while n_success < n_seed:
        seed_i=2022+i*10
        try:
            print("Running for seed %d ...\n"% (seed_i))
            main(seed=seed_i)
            n_success+=1
        except Exception as e:
            print(e)
            pass
        i+=1