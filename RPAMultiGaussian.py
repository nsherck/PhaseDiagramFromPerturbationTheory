#!/usr/bin/env python
import numpy as np
from scipy.optimize import least_squares
import sys, argparse, re
import matplotlib.pyplot as plt

# T075
#a=[8.24741e-01, 2.51015e+00]     # Monomer smearing scale
#u0=[2.54908e+01, -2.38570e+01]     # Excluded-volume parameter
#Temperature = 75
#N = 7
#c_low = 1E-10
#c_high = 100.
#coex_init = [2.91E-9,1.0]
#coex_init = [2.92426958e-07, 3.80062728e-01]

def RPAMultiGaussianFnx(Temperature, N,a_list,u0_list,c_low,c_high,coex_init):
    ''' Generates RPA Phase Diagrams ''' 

    # Debye function for a CGC homopolymer
    def gD_CGC(k2):
        ''' Continuous Gaussian Chain '''
        result1 = 1. - k2/3.0 + k2*k2/12.0 - k2*k2*k2/60.0
        result2 = (2/k2**2*(np.exp(-k2)+k2-1))
        return np.where(k2<0.01,result1,result2)

    def gD_DGC(k2,_N):
        ''' Discrete Gaussian Chain '''
        gD=0.
        for i in range(0, _N+1):
            for j in range(0, _N+1):
                gD = gD + np.exp(-k2*np.abs(i-j)/_N)
        return gD / (N*N + 2*N + 1)

    # Square of smearing function
    def Gaussian(k2,a):
        ''' Fourier transform of Gaussian'''
        return np.exp(-k2*a*a/2)

    # Pressure: integrand of RPA loop integral
    def RPA_Pikernel(_gDGSq,_N,_C):
        Piex_k = _C*_N*_gDGSq/(1. + _N*_C*_gDGSq)
        Piex_k = Piex_k - np.log(1. + _N*_C*_gDGSq)
        return Piex_k

    # Chemical potential: integrand of RPA loop integral
    def RPA_mukernel(_gDGSq,_N,_C):
        muex_k = _N**2*_gDGSq/(1. + _N*_C*_gDGSq)
        return muex_k

    # Intensive Free energy: integrand of RPA loop integral
    def RPA_Fkernel(_gDGSq,_N,_C):
        Fex_k = 1. + _N*_C*_gDGSq
        Fex_k = np.log(Fex_k)
        return Fex_k

    # RPA observables
    def RPA_continuum(a_list,u0_list,_C,_N,UseCGC,_kmin,_kmax,_nkgrid):
        # Generate a large dense 1D mesh of k points
        klist, dk = np.linspace(_kmin, _kmax, _nkgrid, endpoint=True, retstep=True)
        k2list = np.square(klist)

        # build gaussian interactions 
        Gauss_array = np.zeros(klist.size)
        for ng, u0 in enumerate(u0_list):
            prefactor = u0
            Gauss_array = np.add(Gauss_array,prefactor*Gaussian(k2list,a_list[ng]))

        # build the second virial coefficient
        _B2 = 0.
        for ng, u0 in enumerate(u0_list):
            _B2 += u0


        # Form gD*Gaussian for all k
        if UseCGC:
            gDGSq=gD_CGC(k2list)
            np.savetxt("DebyeFunction_CGC.dat",np.transpose([k2list,gDGSq]))
            gDGSq=gDGSq * Gauss_array
            np.savetxt("DebyeFunctionTimesGamma2_CGC.dat",np.transpose([k2list,gDGSq]))
            np.savetxt("Gaussian_DGC_N{}.dat".format(_N),np.transpose([k2list,Gauss_array]))
        else:
            gDGSq=gD_DGC(k2list,_N)
            np.savetxt("DebyeFunction_DGC_N{}.dat".format(_N),np.transpose([k2list,gDGSq]))
            gDGSq=gDGSq * Gauss_array
            np.savetxt("DebyeFunctionTimesGamma2_DGC_N{}.dat".format(_N),np.transpose([k2list,gDGSq]))       
            np.savetxt("Gaussian_DGC_N{}.dat".format(_N),np.transpose([k2list,Gauss_array]))
            
            
            
        #
        FoVig = _C*np.log(_C) - _C
        FoVmft = 0.5*_B2*_C*_C
        FoVex = np.sum(k2list*RPA_Fkernel(gDGSq,_N,_C))/(2.*np.pi)**2*dk
        #
        muig = np.log(_C/_N)
        mumft = _B2*_C*_N
        muex = np.sum(k2list*RPA_mukernel(gDGSq,_N,_C))/(2.*np.pi)**2*dk
        #
        Piig = _C/_N
        Pimft = 0.5*_B2*_C*_C
        Piex = np.sum(k2list*RPA_Pikernel(gDGSq,_N, _C))/(2.*np.pi)**2*dk
        #
        return FoVig+FoVmft+FoVex,FoVig+FoVmft,muig+mumft+muex,muig+mumft,Piig+Pimft+Piex,Piig+Pimft

    #a=[8.24741e-01, 2.51015e+00]     # Monomer smearing scale
    #u0=[2.54908e+01, -2.38570e+01]     # Excluded-volume parameter

    def FindCoexistencePoint(coex_init,a,u0,N,UseCGC,k_max,k_evals):
        ''' Uses least-squares optimization to find coexistence point.
            - The reason for algorithm termination:
                -1 : improper input parameters status returned from MINPACK.
                0 : the maximum number of function evaluations is exceeded.
                1 : gtol termination condition is satisfied.
                2 : ftol termination condition is satisfied.
                3 : xtol termination condition is satisfied.
                4 : Both ftol and xtol termination conditions are satisfied.

        '''
        
        # Coexistence Conditions:
        # P_I == P_II && Mu_I == Mu_II && T_I == T_II 
        
        x0 = coex_init # Initial Guess for Optimization
        lsq_log = open('lsq_log.dat','w')
        def obj(x,a,u0,N,UseCGC,k_max,flag_MuOnly): 
            """Calculate Mu and Pi"""
            F_I,F_mft_I,mu_I,mu_mft_I,Pi_I,Pi_mft_I = RPA_continuum(a,u0,x[0],N,UseCGC,0.,k_max,k_evals)
            F_II,F_mft_II,mu_II,mu_mft_II,Pi_II,Pi_mft_II = RPA_continuum(a,u0,x[1],N,UseCGC,0.,k_max,k_evals)
            weight_P = 1./Pi_I
            weight_mu = 1./mu_I
            #obj_out = weight_P*(Pi_I-Pi_II)+weight_mu*(mu_I-mu_II)
            obj_p  = ((Pi_I-Pi_II))
            obj_mu = ((mu_I-mu_II))
            if flag_MuOnly:
                obj_out = obj_mu
            else:
                obj_out = obj_p + obj_mu
            lsq_log.write('Obj: {}  Obj_P: {}   Obj_Mu: {}  Params: {}\n'.format(obj_out,obj_p,obj_mu,x))
            return [obj_p,obj_mu]
        
        bounds = [[0.,0.],[100.,100.]]
        disableBounds = False
        method='trf'
        
        flag_MuOnly = False
        if method == 'lm' or disableBounds:
            bounds = [[-np.inf,-np.inf],[np.inf,np.inf]]
        elif x0[0] == 1E-15:
            bounds = [[1E-15,0.],[1E-14,np.inf]]
            flag_MuOnly = True
            
        opt = least_squares(obj,x0, args = (a,u0,N,UseCGC,k_max,flag_MuOnly),method=method,bounds=bounds, ftol=1e-13, xtol=1e-13, gtol=1e-13,x_scale='jac',max_nfev=10000)
        c_coex  = opt.x
        cost    = opt.cost
        status  = opt.status
        nfev    = opt.nfev
        
        F_I,F_mft_I,mu_I,mu_mft_I,Pi_I,Pi_mft_I = RPA_continuum(a,u0,c_coex[0],N,UseCGC,0.,k_max,k_evals)
        F_II,F_mft_II,mu_II,mu_mft_II,Pi_II,Pi_mft_II = RPA_continuum(a,u0,c_coex[1],N,UseCGC,0.,k_max,k_evals)
        
        lsq_log.write('Pressure Check: {} {}\n'.format(Pi_I,Pi_II))
        lsq_log.write('Mu Check:       {} {}\n'.format(mu_I,mu_II))
        
        lsq_log.write('\n{}'.format(c_coex))
        lsq_log.write('\nLSQ: {}\n'.format(cost))
        lsq_log.write('\nstatus: {}\n'.format(status))
        lsq_log.write('\nnfev: {}\n'.format(nfev))
        lsq_log.write('\nbounds: {}\n'.format(bounds))
        
        lsq_log.close()

        return c_coex,cost,status,nfev
                 
    a = a_list
    u0 = u0_list
    N=N      
    UseCGC = False # Switch between CGC and DGC
    log_space = True

    # RPA
    if UseCGC:
        filename="MultiGauss_RPA_CGC_T_{}.dat".format(Temperature)
    else:
        filename="MultiGauss_RPA_DGC_N_{}_T_{}.dat".format(N,Temperature)
   
    if log_space:
        C_values = np.logspace(np.log10(c_low),np.log10(c_high),250)
    else:
        C_values = np.linspace(c_low,c_high,100)
    
    # Numerical parameters for the loop integrals 
    k_max = 4*2*np.pi/min(a_list) # pick the max k-value in the list based on smallest length scale
    deltak = 0.1 # the resolution for the loop integrals
    k_evals = k_max/deltak #
    print('k_max: {}'.format(k_max))
    print('k_evals: {}'.format(k_evals))
    
    out=open(filename, 'w')
    out.write("# C Pi(RPA) Pi(MFT) mu(RPA) mu(MFT) F(RPA) F(MFT) k_max: {}\n".format(k_max))
    thermo_list = []
    for C in C_values.tolist():
        #print(C)
        F,F_mft,mu,mu_mft,Pi,Pi_mft = RPA_continuum(a,u0,C,N,UseCGC,0.,k_max,k_evals) # Max k in l units (i.e. b_kuhn/sqrt(6))
        out.write("{} {} {} {} {} {} {}\n".format(C,Pi,Pi_mft,mu,mu_mft,F,F_mft))
        thermo_list.append([C,Pi,Pi_mft,mu,mu_mft,F,F_mft])
    out.close()
    
    thermo_array = np.asarray(thermo_list)
    
    plt.figure()
    plt.plot(thermo_array[:,3],thermo_array[:,1],linewidth = 3)
    #plt.plot(rs,u_gauss,label="{}-Gaussian".format(n),linewidth = 3)
    #plt.scatter(np.linspace(0,rcut,len(knots)),knots,label = "spline knots",c='r')
    plt.ylim(-0.25,0.25)
    plt.xlim(-10.,10.)
    plt.xlabel('mu')
    plt.ylabel('P')
    #plt.legend(loc='best')
    plt.savefig('MuVP.pdf')
    
    plt.figure()
    plt.loglog(thermo_array[:,0],thermo_array[:,1],label='RPA',linewidth = 3)
    plt.loglog(thermo_array[:,0],thermo_array[:,2],label='MFT',linewidth = 3)
    #plt.plot(rs,u_gauss,label="{}-Gaussian".format(n),linewidth = 3)
    #plt.scatter(np.linspace(0,rcut,len(knots)),knots,label = "spline knots",c='r')
    #plt.ylim(min(np.min(u_spline),np.min(u_gauss))*1.25,2)
    #plt.xlim(0,rcut)
    plt.xlabel('C')
    plt.ylabel('P')
    plt.legend(loc='best')
    plt.savefig('PvC.pdf')
    
    plt.figure()
    plt.loglog(thermo_array[:,0],thermo_array[:,3],label='RPA',linewidth = 3)
    plt.loglog(thermo_array[:,0],thermo_array[:,4],label='MFT',linewidth = 3)
    #plt.plot(rs,u_gauss,label="{}-Gaussian".format(n),linewidth = 3)
    #plt.scatter(np.linspace(0,rcut,len(knots)),knots,label = "spline knots",c='r')
    #plt.ylim(min(np.min(u_spline),np.min(u_gauss))*1.25,2)
    #plt.xlim(0,rcut)
    plt.xlabel('C')
    plt.ylabel('Mu')
    plt.legend(loc='best')
    plt.savefig('MuvC.pdf')
    
    FindCoexPt = True
    
    if FindCoexPt:
        c_coex,cost,status,nfev = FindCoexistencePoint(coex_init,a,u0,N,UseCGC,k_max,k_evals)
        
        if np.min(c_coex) < 1E-10: # Repeat with C_I = 0 (i.e. conc ~ 0)
            coex_init = [1.01E-15,np.max(c_coex)]
            c_coex,cost,status,nfev = FindCoexistencePoint(coex_init,a,u0,N,UseCGC,k_max,k_evals)
        
    return c_coex,cost,status,nfev
      
#RPAMultiGaussianFnx(Temperature, N,a,u0,c_low,c_high,coex_init)