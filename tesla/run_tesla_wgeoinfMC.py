"""
Main function to run whitened preconditioned Crank-Nicolson sampling for the time series problem
----------------------
Shiwei Lan @ ASU, 2022
----------------------
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
from sampler.wht_geoinfMC import wht_geoinfMC

np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('alg_NO', nargs='?', type=int, default=0)
    parser.add_argument('seed_NO', nargs='?', type=int, default=2022)
    parser.add_argument('mdl_NO', nargs='?', type=int, default=2)
    parser.add_argument('ker_NO', nargs='?', type=int, default=0)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('num_samp', nargs='?', type=int, default=10000)
    parser.add_argument('num_burnin', nargs='?', type=int, default=5000)
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','bsv','qep'))
    parser.add_argument('kers', nargs='?', type=str, default=('covf','serexp'))
    parser.add_argument('step_sizes', nargs='?', type=float, default=(1e-4,1e-5,1e-5,1e-3,1e-3))
    parser.add_argument('step_nums', nargs='?', type=int, default=[1,1,5,1,5])
    parser.add_argument('algs', nargs='?', type=str, default=('wpCN','winfMALA','winfHMC','winfmMALA','winfmHMC'))
    args = parser.parse_args()
    
    # set random seed
    # seed=args.seed_NO
    np.random.seed(seed)
    
    ## define the linear inverse problem ##
    prior_params={'prior_option':args.mdls[args.mdl_NO],
                  'ker_opt':args.kers[args.ker_NO],#'serexp' if args.mdls[args.mdl_NO]=='bsv' else args.kers[args.ker_NO],
                  'cov_opt':'matern',
                  'basis_opt':'Fourier', # serexp param
                  'KL_trunc':100,
                  'space':'fun',
                  'sigma2':10 if args.mdls[args.mdl_NO]=='gp' else 1,
                  's':2,
                  'q':2 if args.mdls[args.mdl_NO]=='gp' else args.q,
                  'store_eig':True}
    lik_params={'start_date':'2022-01-01',
                'end_date':'2023-01-01'}
    ts = Tesla(**prior_params,**lik_params,seed=seed)
    
    # initialization random noise epsilon
    try:
        # fld=os.path.join(os.getcwd(),'reconstruction/MAP')
        # with open(os.path.join(fld,'MAP.pckl'),'rb') as f:
        #     map=pickle.load(f)
        # f.close()
        # z_init=ts.whiten.qep2wn(map).flatten(order='F')
        u_init=ts.init_parameter if hasattr(ts,'init_parameter') else ts._init_param()#init_opt='LSE',lmda=.1)
        z_init=ts.whiten.qep2wn(u_init).flatten(order='F')
    except Exception as e:
        print(e)
        z_init=ts.whiten.sample()
    h=1e-7; v=np.random.randn(ts.prior.dim)
    l,g=ts.get_geom(z_init,geom_ord=[0,1],whitened=True)[:2]; hess=ts.get_geom(z_init,geom_ord=[1.5],whitened=True, spdapx=True)[2]
    Hv=hess(v)
    l1,g1=ts.get_geom(z_init+h*v,geom_ord=[0,1],whitened=True)[:2]
    print('error in gradient: %0.8f' %(abs((l1-l)/h-g.dot(v))/np.linalg.norm(v)))
    print('error in Hessian: %0.8f' %(np.linalg.norm(-(g1-g)/h-Hv)/np.linalg.norm(v)))
    
    # # adjust the sample size
    # if args.alg_NO>=3:
    #     args.num_samp=5000
    #     args.num_burnin=2000
    
    # run MCMC to generate samples
    print("Running %s sampler with step size %g for %d step(s) for %s prior model with %s kernel taking random seed %d ..." 
          % (args.algs[args.alg_NO],args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO], args.mdls[args.mdl_NO],args.kers[args.ker_NO], seed))
    
    winfMC=wht_geoinfMC(z_init,ts,args.step_sizes[args.alg_NO],args.step_nums[args.alg_NO],args.algs[args.alg_NO],transformation=ts.whiten.wn2qep, MF_only=True, whitened=True, k=100, spdapx=True)
    res=winfMC.sample(args.num_samp,args.num_burnin,return_result=True)#, save_result=False)
    
    # samp=[]; loglik=[]; times=[]
    # accp=0; acpt=0
    # sampler=getattr(winfMC,args.algs[args.alg_NO])
    # prog=np.ceil((args.num_samp+args.num_burnin)*(.05+np.arange(0,1,.05)))
    # beginning=timeit.default_timer()
    # for i in range(args.num_samp+args.num_burnin):
    #     if i==args.num_burnin:
    #         # start the timer
    #         tic=timeit.default_timer()
    #         print('\nBurn-in completed; recording samples now...\n')
    #     # generate MCMC sample with given sampler
    #     acpt_ind,_=sampler()
    #     u,l=winfMC.u,winfMC.ll
    #     # display acceptance at intervals
    #     if i+1 in prog:
    #         print('{0:.0f}% has been completed.'.format(np.float(i+1)/(args.num_samp+args.num_burnin)*100))
    #     # online acceptance rate
    #     accp+=acpt_ind
    #     if (i+1)%100==0:
    #         print('Acceptance at %d iterations: %0.2f' % (i+1,accp/100))
    #         accp=0.0
    #     # save results
    #     loglik.append(l)
    #     if i>=args.num_burnin:
    #         samp.append(T(u))
    #         acpt+=acpt_ind
    #     times.append(timeit.default_timer()-beginning)
    # # stop timer
    # toc=timeit.default_timer()
    # time_=toc-tic
    # acpt/=args.num_samp
    # print("\nAfter %g seconds, %d samples have been collected. \n" % (time_,args.num_samp))
    #
    # # store the results
    # samp=np.stack(samp); loglik=np.stack(loglik);  times=np.stack(times)
    # # name file
    # ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    # savepath=os.path.join(os.getcwd(),'result')
    # if not os.path.exists(savepath): os.makedirs(savepath)
    # filename='Tesla_'+str(ts.misfit.size)+'days_'+args.algs[args.alg_NO]+'_dim'+str(len(u))+'_'+prior_params['prior_option']+'_'+prior_params['ker_opt']+'_'+ctime+'.pckl'
    # f=open(os.path.join(savepath,filename),'wb')
    # pickle.dump([prior_params, lik_params, args, samp,loglik,time_,times],f)
    # f.close()

if __name__ == '__main__':
    # main()
    n_seed = 10; i=0; n_success=0
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