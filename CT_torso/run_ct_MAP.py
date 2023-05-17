"""
Main function to obtain maximum a posterior (MAP) for linear inverse problem of torso CT
----------------------
Shiwei Lan @ ASU, 2022
"""

# modules
import os,argparse,pickle
import numpy as np
from scipy import optimize
import timeit,time

# the inverse problem
from CT import CT

np.set_printoptions(precision=3, suppress=True)
# np.random.seed(2022)

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('mdl_NO', nargs='?', type=int, default=2)
    parser.add_argument('ker_NO', nargs='?', type=int, default=1)
    parser.add_argument('q', nargs='?', type=int, default=1)
    parser.add_argument('whiten', nargs='?', type=int, default=0) # choose to optimize in white noise representation space
    parser.add_argument('NCG', nargs='?', type=int, default=0) # choose to optimize with Newton conjugate gradient method
    parser.add_argument('mdls', nargs='?', type=str, default=('gp','bsv','qep'))
    parser.add_argument('kers', nargs='?', type=str, default=('covf','serexp','graphL'))
    args = parser.parse_args()
    
    # set random seed
    np.random.seed(seed)
    
    ## define the linear inverse problem ##
    prior_params={'prior_option':args.mdls[args.mdl_NO],
                  'ker_opt':'serexp' if args.mdls[args.mdl_NO]=='bsv' else args.kers[args.ker_NO],
                  # 'ker_opt':{'gp':'graphL','bsv':'serexp','qep':args.kers[args.ker_NO]}[args.mdls[args.mdl_NO]],
                  'basis_opt':'Fourier', # serexp param
                  'KL_trunc':5000,
                  'space':'vec' if args.kers[args.ker_NO]!='graphL' else 'fun',
                  'sigma2':1e3,
                  's':2 if args.mdls[args.mdl_NO]=='gp' else 1,
                  'q':2 if args.mdls[args.mdl_NO]=='gp' else args.q,
                  'store_eig':args.kers[args.ker_NO]!='graphL',
                  'normalize':True, # graphL param
                  'weightedge':True} # graphL param
    lik_params={'CT_set':'proj200_loc512',
                'data_set':'torso'}
    ct = CT(**prior_params,**lik_params,seed=seed)
    truth = ct.misfit.truth
    
    # optimze
    print("Obtaining MAP estimate for %s prior model with %s kernel %s ..." % ({0:'Gaussian',1:'Besov',2:'q-Exponential'}[args.mdl_NO], args.kers[args.ker_NO], {True:'using Newton CG',False:''}[args.NCG]))
    
    if not hasattr(ct,'init_parameter'): ct._init_param()#init_opt='LSE',lmda=.1)
    param0 = ct.init_parameter
    if args.whiten: param0 = ct.whiten.qep2wn(param0).flatten(order='F')
    fun = lambda parameter: ct._get_misfit(ct.whiten.wn2qep(parameter) if args.whiten else parameter, MF_only=False, incldet=False)
    def grad(parameter):
        param = ct.whiten.wn2qep(parameter) if args.whiten else parameter
        g = ct._get_grad(param, MF_only=False)
        if args.whiten: g = ct.whiten.wn2qep(parameter, 1)(g, adj=True)
        return g.squeeze()
    def hessp(parameter,v):
        param = ct.whiten.wn2qep(parameter) if args.whiten else parameter
        Hv = ct._get_HessApply(param, MF_only=False)(ct.whiten.wn2qep(parameter,1)(v) if args.whiten else v)
        if args.whiten:
            Hv = ct.whiten.wn2qep(parameter, 1)(Hv, adj=True) 
            Hv+= ct.whiten.wn2qep(parameter, 2)(v, ct._get_grad(param, MF_only=False), adj=True)
        return Hv.squeeze()
    # h=1e-7; v=ct.whiten.sample() if args.whiten else ct.prior.sample()
    # # if args.whiten: v=ct.whiten.qep2wn(v).flatten(order='F')
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
        Xi_=ct.whiten.wn2qep(Xi) if args.whiten else Xi
        ERR.append(np.linalg.norm((ct.prior.vec2fun(Xi_) if ct.prior.space=='vec' else Xi_) -truth.flatten(order='F'))/np.linalg.norm(truth))
    print('{0:4s}   {1:9s}   {2:9s}   {3:9s}   {4:9s}'.format('Iter', ' X1', ' X2', ' X3', 'f(X)'))
    # solve for MAP
    start = time.time()
    if args.NCG:
        res = optimize.minimize(fun, param0, method='trust-ncg', jac=grad, hessp=hessp, callback=call_back, options={'maxiter':100,'disp':True})
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
    map_v=ct.whiten.wn2qep(res.x) if args.whiten else res.x
    map_f=ct.prior.vec2fun(map_v) if ct.prior.space=='vec' else map_v; funs=np.stack(FUN); errs=np.stack(ERR)
    print('Relative error of MAP compared with the truth %.2f%%' % (errs[-1]*100))
    map_f=map_f.reshape(ct.misfit.size,order='F')
    # name file
    ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
    savepath=os.path.join(os.getcwd(),'MAP')
    if not os.path.exists(savepath): os.makedirs(savepath)
    # save
    filename='ct_MAP_dim'+str(len(param0))+'_'+prior_params['prior_option']+'_'+prior_params['ker_opt']+('_whiten' if args.whiten else '')+('_NCG' if args.NCG else '')+'_'+ctime+'.pckl'
    f=open(os.path.join(savepath,filename),'wb')
    pickle.dump([prior_params, truth, map_f, funs, errs],f)
    f.close()
    
    # plot
    ct.misfit.plot_data(img=map_f, save_img=True, save_path=savepath, save_fname=prior_params['prior_option']+'_'+prior_params['ker_opt']+('_whiten' if args.whiten else '')+('_NCG' if args.NCG else ''))

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