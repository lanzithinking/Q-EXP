#!/usr/bin/env python
"""
Whitened Geometric Infinite dimensional MCMC samplers
using FEniCS dolfin library https://bitbucket.org/fenics-project/dolfin
Shiwei Lan @ ASU, 2023
-------------------------------
Based on "Dimension-Robust MCMC samplers"
Victor Chen, Matthew M. Dunlop, Omiros Papaspiliopoulos, Andrew M. Stuart
https://arxiv.org/abs/1803.03344
-----------------------------------
Created October 11, 2016
-----------------------------------
Modified May 13, 2023 @ ASU
"""
__author__ = "Shiwei Lan"
__copyright__ = "Copyright 2022, The STBP project"
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Shiwei Lan"
__email__ = "slan@asu.edu; lanzithinking@outlook.com;"

import numpy as np
import timeit,time
import dolfin as df

class wht_geoinfMC:
    """
    Whitened version of 
    Geometric Infinite dimensional MCMC samplers by Beskos, Alexandros, Mark Girolami, Shiwei Lan, Patrick E. Farrell, and Andrew M. Stuart.
    https://www.sciencedirect.com/science/article/pii/S0021999116307033
    -------------------------------------------------------------------
    It has been configured to use low-rank Hessian approximation.
    After the class is instantiated with arguments, call sample to collect MCMC samples which will be stored in 'result' folder.
    """
    def __init__(self,parameter_init,model,step_size,step_num,alg_name,adpt_h=False,**kwargs):
        """
        Initialization
        --------------
        u: parameter to be sampled
        dim: dimension of the parameter
        model: an object with properties including misfit (negative loglikelihood), prior and posterior (Gaussian approximation)
        T: a transformation that maps white noise to the (non-Gaussian) prior distribution
        geom: the method that outputs geometric quantities including log-likelihood(posterior), gradient, Hessian and its eigenvalues
        geomT: geom o T
        """
        # parameters
        self.u=parameter_init
        self.dim=self.u.size()
        self.model=model
        self.T=kwargs.pop('transformation',None)
        
        target_acpt=kwargs.pop('target_acpt',0.65)
        # geometry needed
        geom_ord=[0]
        if any(s in alg_name for s in ['MALA','HMC']): geom_ord.append(1)
        if any(s in alg_name for s in ['mMALA','mHMC']): geom_ord.append(2)
        geomf=kwargs.pop('geomT',None) if 'geomT' in kwargs else getattr(model,'get_geom')
        self.geom=lambda parameter: geomf(parameter,geom_ord=geom_ord,**kwargs)
        self.ll,self.g,_,self.eigs=self.geom(self.u)

        # sampling setting
        self.h=step_size
        self.L=step_num
        if 'HMC' not in alg_name: self.L=1
        self.alg_name = alg_name

        # optional setting for adapting step size
        self.adpt_h=adpt_h
        if self.adpt_h:
            h_adpt={}
#             h_adpt['h']=self._init_h()
            h_adpt['h']=self.h
            h_adpt['mu']=np.log(10*h_adpt['h'])
            h_adpt['loghn']=0.
            h_adpt['An']=0.
            h_adpt['gamma']=0.05
            h_adpt['n0']=10
            h_adpt['kappa']=0.75
            h_adpt['a0']=target_acpt
            self.h_adpt=h_adpt
    
    def randv(self):
        """
        sample v ~ N(0,I) or N(0,invK(u))
        """
        if any(s in self.alg_name for s in ['mMALA','mHMC']) and hasattr(self.model, 'post_Ga'):
            v_ = self.model.whtprior.sample()
            v = v_.copy(); v.zero()
            self.model.post_Ga.sample(v_, v, add_mean=False)
        elif hasattr(self.model, 'whtprior'):
            v = self.model.whtprior.sample()
        else:
            raise Error('No matching noise sampling!')
        return v
        
    def wpCN(self):
        """
        Whitened preconditioned Crank-Nicolson
        """
        # initialization
        u=self.u.copy()
        
        # sample velocity
        v=self.randv()

        # generate proposal according to Crank-Nicolson scheme
        u.zero()
        u.axpy((1-self.h/4)/(1+self.h/4),self.u)
        u.axpy(np.sqrt(self.h)/(1+self.h/4),v)

        # update geometry
        ll=self.geom(u)[0]

        # Metropolis test
        logr=ll-self.ll

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.u=u; self.ll=ll;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr

    def winfMALA(self):
        """
        Whitened infinite dimensional Metropolis Adjusted Langevin Algorithm
        """
        # initialization
        u=self.u.copy()
        rth=np.sqrt(self.h)
        
        # sample velocity
        v=self.randv()

        # natural gradient
        ng=self.model.whtprior.C_act(self.g)

        # update velocity
        v.axpy(rth/2,ng)

        # current energy
        E_cur = -self.ll - rth/2*self.g.inner(v) + self.h/8*self.g.inner(ng)

        # generate proposal according to Langevin dynamics
        u.zero()
        u.axpy((1-self.h/4)/(1+self.h/4),self.u)
        u.axpy(rth/(1+self.h/4),v)

        # update velocity
        v_=v.copy();v.zero()
        v.axpy(-(1-self.h/4)/(1+self.h/4),v_)
        v.axpy(rth/(1+self.h/4),self.u)

        # update geometry
        ll,g=self.geom(u)[:2]

        # natural gradient
        ng=self.model.whtprior.C_act(g)

        # new energy
        E_prp = -ll - rth/2*g.inner(v) + self.h/8*g.inner(ng)

        # Metropolis test
        logr=-E_prp+E_cur

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.u=u; self.ll=ll; self.g=g;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr

    def winfHMC(self):
        """
        Whitened infinite dimensional Hamiltonian Monte Carlo
        """
        # initialization
        u=self.u.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA
        cos_=np.cos(rth); sin_=np.sin(rth);

        # sample velocity
        v=self.randv()

        # natural gradient
        ng=self.model.whtprior.C_act(self.g)

        # accumulate the power of force
        pw = rth/2*self.g.inner(v)

        # current energy
        E_cur = -self.ll - self.h/8*self.g.inner(ng)

        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v.axpy(rth/2,ng)

            # a full step for position
            u_=u.copy();u.zero()
            u.axpy(cos_,u_)
            u.axpy(sin_,v)
            v_=v.copy();v.zero()
            v.axpy(-sin_,u_)
            v.axpy(cos_,v_)

            # update geometry
            ll,g,_,_=self.geom(u)
            ng=self.model.whtprior.C_act(g)

            # another half step for velocity
            v.axpy(rth/2,ng)

            # accumulate the power of force
            if l!=randL-1: pw+=rth*g.inner(v)

        # accumulate the power of force
        pw += rth/2*g.inner(v)

        # new energy
        E_prp = -ll - self.h/8*g.inner(ng)

        # Metropolis test
        logr=-E_prp+E_cur-pw

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.u=u; self.ll=ll; self.g=g;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr

    def winfmMALA(self):
        """
        Whitened infinite dimensional manifold MALA
        """
        # initialization
        u=self.u.copy()
        rth=np.sqrt(self.h)
        
        # sample velocity
        v=self.randv()

        # natural gradient
        ng=self.model.post_Ga.postC_act(self.g) # use low-rank posterior Hessian solver

        # update velocity
        v.axpy(rth/2,ng)

        # current energy
        E_cur = -self.ll - rth/2*self.g.inner(v) + self.h/8*self.g.inner(ng) +0.5*self.model.post_Ga.Hlr.inner(v,v) -0.5*sum(np.log(1+self.eigs[0])) # use low-rank Hessian inner product

        # generate proposal according to simplified manifold Langevin dynamics
        u.zero()
        u.axpy((1-self.h/4)/(1+self.h/4),self.u)
        u.axpy(rth/(1+self.h/4),v)

        # update velocity
        v_=v.copy();v.zero()
        v.axpy(-(1-self.h/4)/(1+self.h/4),v_)
        v.axpy(rth/(1+self.h/4),self.u)

        # update geometry
        ll,g,_,eigs=self.geom(u)
        self.model.post_Ga.eigs=eigs # update the eigen-pairs in low-rank approximation --important!

        # natural gradient
        ng=self.model.post_Ga.postC_act(g)

        # new energy
        E_prp = -ll - rth/2*g.inner(v) + self.h/8*g.inner(ng) +0.5*self.model.post_Ga.Hlr.inner(v,v) -0.5*sum(np.log(1+eigs[0]))

        # Metropolis test
        logr=-E_prp+E_cur

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.u=u; self.ll=ll; self.g=g; self.eigs=eigs;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr

    def winfmHMC(self):
        """
        Whitened infinite dimensional manifold HMC
        """
        # initialization
        u=self.u.copy()
        rth=np.sqrt(self.h) # make the scale comparable to MALA
#         cos_=np.cos(rth); sin_=np.sin(rth);
        cos_=(1-self.h/4)/(1+self.h/4); sin_=rth/(1+self.h/4);

        # sample velocity
        v=self.randv()

        # natural gradient
        ng=self.model.post_Ga.postC_act(self.g) # use low-rank posterior Hessian solver

        # accumulate the power of force
        pw = rth/2*self.model.whtprior.C_act(v,-1).inner(ng)

        # current energy
        E_cur = -self.ll - self.h/4*self.model.whtprior.cost(ng) +0.5*self.model.post_Ga.Hlr.inner(v,v) -0.5*sum(np.log(1+self.eigs[0])) # use low-rank Hessian inner product

        randL=np.int(np.ceil(np.random.uniform(0,self.L)))

        for l in range(randL):
            # a half step for velocity
            v.axpy(rth/2,ng)

            # a full step rotation
            u_=u.copy();u.zero()
            u.axpy(cos_,u_)
            u.axpy(sin_,v)
            v_=v.copy();v.zero()
            v.axpy(cos_,v_)
            v.axpy(-sin_,u_)

            # update geometry
            ll,g,_,eigs=self.geom(u)
            self.model.post_Ga.eigs=eigs # update the eigen-pairs in low-rank approximation --important!
            ng=self.model.post_Ga.postC_act(g)

            # another half step for velocity
            v.axpy(rth/2,ng)

            # accumulate the power of force
            if l!=randL-1: pw+=rth*self.model.whtprior.C_act(v,-1).inner(ng)

        # accumulate the power of force
        pw += rth/2*self.model.whtprior.C_act(v,-1).inner(ng)

        # new energy
        E_prp = -ll - self.h/4*self.model.whtprior.cost(ng) +0.5*self.model.post_Ga.Hlr.inner(v,v) -0.5*sum(np.log(1+eigs[0]))

        # Metropolis test
        logr=-E_prp+E_cur-pw

        if np.isfinite(logr) and np.log(np.random.uniform())<min(0,logr):
            # accept
            self.u=u; self.ll=ll; self.g=g; self.eigs=eigs;
            acpt=True
        else:
            acpt=False

        # return accept indicator
        return acpt,logr
    
    def _init_h(self):
        """
        find a reasonable initial step size
        """
        h=1.
        _self=self
        sampler=getattr(_self,str(_self.alg_name))
        _self.h=h;_self.L=1
        _,logr=sampler()
        a=2.*(np.exp(logr)>0.5)-1.
        while a*logr>-a*np.log(2):
            h*=pow(2.,a)
            _self=self
            _self.h=h;_self.L=1
            _,logr=sampler()
        return h
    
    def _dual_avg(self,iter,an):
        """
        dual-averaging to adapt step size
        """
        hn_adpt=self.h_adpt
        hn_adpt['An']=(1.-1./(iter+hn_adpt['n0']))*hn_adpt['An'] + (hn_adpt['a0']-an)/(iter+hn_adpt['n0'])
        logh=hn_adpt['mu'] - np.sqrt(iter)/hn_adpt['gamma']*hn_adpt['An']
        hn_adpt['loghn']=pow(iter,-hn_adpt['kappa'])*logh + (1.-pow(iter,-hn_adpt['kappa']))*hn_adpt['loghn']
        hn_adpt['h']=np.exp(logh)
        return hn_adpt
    
    # sample with given method
    def sample(self,num_samp,num_burnin,num_retry_bad=0,**kwargs):
        """
        sample with given MCMC method
        """
        name_sampler = str(self.alg_name)
        try:
            sampler = getattr(self, name_sampler)
        except AttributeError:
            print(self.alg_name, 'not found!')
        else:
            print('\nRunning '+self.alg_name+' now...\n')

        # allocate space to store results
        import os
        samp_fname='_samp_'+self.alg_name+'_dim'+str(self.dim)+'_'+time.strftime("%Y-%m-%d-%H-%M-%S")
        samp_fpath=os.path.join(os.getcwd(),'result')
        if not os.path.exists(samp_fpath):
            os.makedirs(samp_fpath)
#         self.samp=df.File(os.path.join(samp_fpath,samp_fname+".xdmf"))
        self.samp=df.HDF5File(self.model.pde.mpi_comm,os.path.join(samp_fpath,samp_fname+".h5"),"w")
        self.loglik=np.zeros(num_samp+num_burnin)
        self.acpt=0.0 # final acceptance rate
        self.times=np.zeros(num_samp+num_burnin) # record the history of time used for each sample
        
        # number of adaptations for step size
        if self.adpt_h:
            self.h_adpt['n_adpt']=kwargs.pop('adpt_steps',num_burnin)
        
        # online parameters
        accp=0.0 # online acceptance
        num_cons_bad=0 # number of consecutive bad proposals

        beginning=timeit.default_timer()
        for s in range(num_samp+num_burnin):

            if s==num_burnin:
                # start the timer
                tic=timeit.default_timer()
                print('\nBurn-in completed; recording samples now...\n')

            # generate MCMC sample with given sampler
            while True:
                try:
                    acpt_idx,logr=sampler()
                except Exception as e:
                    print(e)
#                     import pydevd; pydevd.settrace()
#                     import traceback; traceback.print_exc()
                    if num_retry_bad==0:
                        acpt_idx=False; logr=-np.inf
                        print('Bad proposal encountered! Passing... bias introduced.')
                        break # reject bad proposal: bias introduced
                    else:
                        num_cons_bad+=1
                        if num_cons_bad<num_retry_bad:
                            print('Bad proposal encountered! Retrying...')
                            continue # retry until a valid proposal is made
                        else:
                            acpt_idx=False; logr=-np.inf # reject it and keep going
                            num_cons_bad=0
                            print(str(num_retry_bad)+' consecutive bad proposals encountered! Passing...')
                            break # reject it and keep going
                else:
                    num_cons_bad=0
                    break

            accp+=acpt_idx

            # display acceptance at intervals
            if (s+1)%100==0:
                print('\nAcceptance at %d iterations: %0.2f' % (s+1,accp/100))
                accp=0.0

            # save results
            self.loglik[s]=self.ll
            if s>=num_burnin:
                u_f=df.Function(self.model.prior.V)
                u_f.vector().set_local(self.T(self.u) if callable(self.T) else self.u)
                u_f.vector().apply('insert')
                self.samp.write(u_f,'sample_{0}'.format(s-num_burnin))
                self.acpt+=acpt_idx
            
            # record the time
            self.times[s]=timeit.default_timer()-beginning
            
            # adapt step size h if needed
            if self.adpt_h:
                if s<self.h_adpt['n_adpt']:
                    self.h_adpt=self._dual_avg(s+1,np.exp(min(0,logr)))
                    self.h=self.h_adpt['h']
                    print('New step size: %.2f; \t New averaged step size: %.6f\n' %(self.h_adpt['h'],np.exp(self.h_adpt['loghn'])))
                if s==self.h_adpt['n_adpt']:
                    self.h_adpt['h']=np.exp(self.h_adpt['loghn'])
                    self.h=self.h_adpt['h']
                    print('Adaptation completed; step size freezed at:  %.6f\n' % self.h_adpt['h'])

        # stop timer
        self.samp.close()
        toc=timeit.default_timer()
        self.time=toc-tic
        self.acpt/=num_samp
        print("\nAfter %g seconds, %d samples have been collected with the final acceptance rate %0.2f \n"
              % (self.time,num_samp,self.acpt))

        # save to file
        self.save_samp()

    # save samples
    def save_samp(self):
        import os,errno
        import pickle
        # create folder
        cwd=os.getcwd()
        self.savepath=os.path.join(cwd,'result')
        try:
            os.makedirs(self.savepath)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
            else:
                raise
        # name file
        ctime=time.strftime("%Y-%m-%d-%H-%M-%S")
        self.filename=self.alg_name+'_dim'+str(self.dim)+'_'+ctime
        # dump data
        f=open(os.path.join(self.savepath,self.filename+'.pckl'),'wb')
        res2save=[self.h,self.L,self.alg_name,self.loglik,self.acpt,self.time,self.times]
        if self.adpt_h:
            res2save.append(self.h_adpt)
        pickle.dump(res2save,f)
        f.close()
#         # load data
#         f=open(os.path.join(self.savepath,self.filename+'.pckl'),'rb')
#         f_read=pickle.load(f)
#         f.close()
