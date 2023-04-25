"""
Main function to obtain maximum a posterior (MAP) for the time series problem with some data held out
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
from prior import *

np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('mdl_NO', nargs='?', type=int, default=2)
    parser.add_argument('ker_NO', nargs='?', type=int, default=0)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('whiten', nargs='?', type=int, default=0) # choose to optimize in white noise representation space
    parser.add_argument('NCG', nargs='?', type=int, default=0) # choose to optimize with Newton conjugate gradient method
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
                  'sigma2':1,#000,# if args.mdls[args.mdl_NO]=='gp' else 100,
                  's':1,
                  'q':2 if args.mdls[args.mdl_NO]=='gp' else args.q,
                  'store_eig':True}
    lik_params={'start_date':'2022-01-01',
                'end_date':'2023-01-01'}
    ts = Tesla(**prior_params,**lik_params,seed=seed)
    dat = ts.misfit.obs
    
    # train index
    # tr_idx = np.hstack([np.arange(int(ts.misfit.size/2),dtype=int),int(ts.misfit.size/2)+np.arange(0,int(ts.misfit.size/8*3),2,dtype=int)])
    tr_idx = np.hstack([np.arange(0,int(ts.misfit.size/2),2,dtype=int),
                        int(ts.misfit.size/2)+np.arange(0,int(ts.misfit.size/4),4,dtype=int),
                        int(ts.misfit.size/2)+int(ts.misfit.size/4)+np.arange(0,int(ts.misfit.size/4),8,dtype=int)])
    ts.misfit.size=len(tr_idx)
    ts.misfit.times=ts.misfit.times[tr_idx]
    ts.misfit.obs=ts.misfit.obs[tr_idx]
    if np.size(ts.misfit.nzvar)>1: ts.misfit.nzvar=ts.misfit.nzvar[tr_idx]
    ts.prior=prior(input=ts.misfit.times,**prior_params)
    dat = ts.misfit.obs
    
    # optimize
    print("Obtaining MAP estimate for %s prior model with %s kernel %s ..." % ({0:'Gaussian',1:'Besov',2:'q-Exponential'}[args.mdl_NO], args.kers[args.ker_NO], {True:'using Newton CG',False:''}[args.NCG]))
    
    if not hasattr(ts,'init_parameter'): ts._init_param()
    param0 = ts.init_parameter
    if args.whiten: param0 = ts.whiten.qep2wn(param0).flatten(order='F')
    fun = lambda parameter: ts._get_misfit(ts.whiten.wn2qep(parameter) if args.whiten else parameter, MF_only=False, incldet=False)
    def grad(parameter):
        param = ts.whiten.wn2qep(parameter) if args.whiten else parameter
        g = ts._get_grad(param, MF_only=False)
        if args.whiten: g = ts.whiten.wn2qep(parameter, 1)(g, adj=True)
        return g.squeeze()
    def hessp(parameter,v):
        param = ts.whiten.wn2qep(parameter) if args.whiten else parameter
        Hv = ts._get_HessApply(param, MF_only=False)(ts.whiten.wn2qep(parameter,1)(v) if args.whiten else v)
        if args.whiten:
            Hv = ts.whiten.wn2qep(parameter, 1)(Hv, adj=True) 
            Hv+= ts.whiten.wn2qep(parameter, 2)(v, ts._get_grad(param, MF_only=False), adj=True)
        return Hv.squeeze()
    # h=1e-7; v=ts.whiten.sample() if args.whiten else ts.prior.sample()
    # # if args.whiten: v=ts.whiten.qep2wn(v).flatten(order='F')
    # f,g,Hv=fun(param0),grad(param0),hessp(param0,v)
    # f1,g1=fun(param0+h*v),grad(param0+h*v)
    # print('error in gradient: %0.8f' %(abs((f1-f)/h-g.dot(v))/np.linalg.norm(v)))
    # print('error in Hessian: %0.8f' %(np.linalg.norm((g1-g)/h-Hv)/np.linalg.norm(v)))
    
    global Nfeval,FUN,ERR
    Nfeval=1; FUN=[]; ERR=[];
    def call_back(Xi):
        global Nfeval,FUN,ERR
        fval=fun(Xi)
        print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], fval))
        Nfeval += 1
        FUN.append(fval)
        Xi_=ts.whiten.wn2qep(Xi) if args.whiten else Xi
        ERR.append(np.linalg.norm((ts.prior.vec2fun(Xi) if ts.prior.space=='vec' else Xi) -dat.flatten())/np.linalg.norm(dat))
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
    # solve for MAP
    start = time.time()
    if args.NCG:
        res = optimize.minimize(fun, param0, method='trust-ncg', jac=grad, hessp=hessp, callback=call_back, options={'maxiter':1000,'disp':True})
    else:
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
    map_v=ts.whiten.wn2qep(res.x) if args.whiten else res.x
    map_f=ts.prior.vec2fun(map_v) if ts.prior.space=='vec' else map_v; funs=np.stack(FUN); errs=np.stack(ERR)
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'MAP_hldout')
    if not os.path.exists(savepath): os.makedirs(savepath)
    # save
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