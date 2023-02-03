"""
Main function to obtain maximum a posterior (MAP) for the time series problem
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
from scipy import optimize
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
    parser.add_argument('mdl_NO', nargs='?', type=int, default=1)
    parser.add_argument('ker_NO', nargs='?', type=int, default=0)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','bsv','qep'))
    parser.add_argument('kers', nargs='?', type=str, default=('covf','serexp'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(seed)
    
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
    ts = Tesla(**prior_params,**lik_params,seed=seed)
    dat = ts.misfit.obs
    
    # optimize
    print("Obtaining MAP estimate for %s prior model with %s kernel ..." % ({0:'Gaussian',1:'Besov',2:'q-Exponential'}[args.mdl_NO], args.kers[args.ker_NO]))
    
    # param0 = ts.misfit.obs.flatten()
    param0 = ts.prior.sample()
    if ts.prior.space=='vec': param0=ts.prior.fun2vec(param0)
    fun = lambda parameter: ts._get_misfit(parameter, MF_only=False, incldet=False)
    grad = lambda parameter: ts._get_grad(parameter, MF_only=False)
    global Nfeval,FUN,ERR
    Nfeval=1; FUN=[]; ERR=[];
    def call_back(Xi):
        global Nfeval,FUN,ERR
        fval=fun(Xi)
        print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], fval))
        Nfeval += 1
        FUN.append(fval)
        ERR.append(np.linalg.norm((ts.prior.vec2fun(Xi) if ts.prior.space=='vec' else Xi) -dat.flatten())/np.linalg.norm(dat))
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
    # solve for MAP
    start = time.time()
    res = optimize.minimize(fun, param0, method='L-BFGS-B' if args.kers[args.ker_NO]=='graphL' else 'BFGS', jac=grad, callback=call_back, options={'maxiter':1000,'disp':True})
    end = time.time()
    print('\nTime used is %.4f' % (end-start))
    # print out info
    if res.success:
        print('\nConverged in ', res.nit, ' iterations.')
    else:
        print('\nNot Converged.')
    print('Final function value: %.4f.\n' % res.fun)
    
    
    # store the results
    MAP=ts.prior.vec2fun(res.x) if ts.prior.space=='vec' else res.x; funs=np.stack(FUN); errs=np.stack(ERR)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'MAP')
    if not os.path.exists(savepath): os.makedirs(savepath)
    filename='Tesla_'+str(ts.misfit.size)+'days_MAP_dim'+str(len(MAP))+'_'+prior_params['prior_option']+'_'+prior_params['ker_opt']+'_'+ctime+'.pckl'
    f=open(os.path.join(savepath,filename),'wb')
    pickle.dump([dat, MAP.reshape(ts.misfit.size), funs, errs],f)
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