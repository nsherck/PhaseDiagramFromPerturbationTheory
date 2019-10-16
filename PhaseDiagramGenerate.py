import time
import numpy as np
import scipy as sp
import scipy.stats
import math
import subprocess as prcs
import os, sys
from shutil import copyfile
import ast
from RPAMultiGaussian import RPAMultiGaussianFnx


#=================
''' USER-INPUT '''
ForceFieldFileDir 	= 'ForceFieldDirectory'
coex_init = [0.2E-9,1.0]
c_low = 1E-15
c_high = 10.

# This can be anything, typically temperature or u0 strengths
param_name_0  = 'T'
FFSweepList_0 = [160]

param_name_1  = 'N'
FFSweepList_1 = [5]

''' ************** Bulk of the Code ************************************* '''
''' ********************************************************************* '''

### FUNCTION DEFINITIONS USED BELOW ###

#CGSweepFTS.GeneratePolyFTSScript('CG_run_LSQFit_OptALL_Final_ff.dat',71.302)

def GeneratePolyFTSScript(force_field_file_name):
    ''' Builds the Gaussian interactions from the force-field file output by sim. '''
    force_field_file = open(force_field_file_name,'r')
    data = force_field_file.read()
    data = data.split('>>> POTENTIAL')
    
    fout = open('PotentialConversions.log','w')
    
    list_potential_names        = []
    list_MD_gauss_potentials    = []
    BondConstant = 0.
    number_gaussians     = 0
    
    for index,potential in enumerate(data):
        if len(potential) <= 0:
            pass
        else:
        
            name = potential.split('{')[0].lstrip()
            list_potential_names.append(name.rstrip())
            pot_dictionary = potential.split('{')[1]
            pot_dictionary = ast.literal_eval('{ '+pot_dictionary)
            if 'Bond' in name:
                BondConstant = pot_dictionary.get('FConst')
                
            if 'Bond' not in name:
                number_gaussians += 1
                MD_B = pot_dictionary.get('B')
                MD_Kappa = pot_dictionary.get('Kappa')
                list_MD_gauss_potentials.append([MD_B,MD_Kappa])
    
    fout.write('Potential Names in Force-field\n')
    fout.write('{}\n'.format(list_potential_names))
    fout.write('The MD parameters in the {} Gaussian Potentials:\n'.format(number_gaussians))
    fout.write('{}\n'.format(list_MD_gauss_potentials))
    fout.write('The force constant: \n')
    fout.write('{}\n'.format(BondConstant))
    
    # Kuhn_Length/sqrt(6) 
    Kuhn_Length = np.sqrt(1./4./BondConstant)
    fout.write('\nThe Kuhn_Length/sqrt(6):\n')
    fout.write('{}\n'.format(Kuhn_Length))
    
    
    list_FTS_gauss_potentials = []
    list_FTS_alphas = []
    list_FTS_u0     = []
    for g,md_gauss in enumerate(list_MD_gauss_potentials):
        kappa       = md_gauss[1]
        FTS_alpha   = np.sqrt(1./2./kappa)
        B           = md_gauss[0]
        FTS_u       = B*(2*math.pi*FTS_alpha**2)**(3./2.)
        # convert to kuhn_length/sqrt(6)
        FTS_alpha   = FTS_alpha/Kuhn_Length
        FTS_u       = FTS_u/(Kuhn_Length)**3
        fout.write('FTS Gaussian Potential Number {}\n'.format(g))
        fout.write('alpha:\n')
        fout.write('{}\n'.format(FTS_alpha))
        fout.write('interaction strength:\n')
        fout.write('{}\n'.format(FTS_u))
        list_FTS_gauss_potentials.append([FTS_u,FTS_alpha])
        list_FTS_alphas.append(FTS_alpha)
        list_FTS_u0.append(FTS_u)
    
    fout.close()
    
    return list_FTS_u0, list_FTS_alphas, number_gaussians
            
  
def CreateCGModelDirectory(RunDirName,param_name_0,param_name_1,Val0,Val1,cwd,ForceFieldFileDir,c_low,c_high,coex_init0):
    ''' Main function for creating the CG directory 

    '''

   # make run dir.
    os.mkdir(RunDirName)
    # copy FF's to run dir.
    source = os.path.join(cwd,ForceFieldFileDir)
    #print (source)
    for subdir, dirs, files in os.walk(source): # Copy the appropriate force-field into run directory
        for file in files:
            #print (file)
            if file.endswith("ff.dat"):
                if str(Val0) in file:
                    copyfile(os.path.join(cwd,ForceFieldFileDir,file),os.path.join(cwd,RunDirName,file))
                    force_field_file_name = file
                    #print('force field file name:')
                    #print('{}'.format(force_field_file_name))

    # move into new directory
    os.chdir(RunDirName)  
    
    # Calculate the gaussian interactions from the Srel force-field file
    u0_list, a_list, ng = GeneratePolyFTSScript(force_field_file_name)
   
    a   = a_list     # Monomer smearing scale
    u0  = u0_list    # Excluded-volume parameter

    Temperature = Val0
    N = Val1
    c_low = c_low
    c_high = c_high
    coex_init0 = coex_init0

    c_coex,cost,status,nfev = RPAMultiGaussianFnx(Temperature,N,a,u0,c_low,c_high,coex_init0)
    
    # Move backup one directory
    os.chdir("..")
   
    return c_coex,cost,status,nfev
    
''' ********************************************************************************* '''
''' ********* THE CODE THAT CALLS THE ABOVE FUNCTIONS TO GENERATE CG RUNS *********** '''
''' ********************************************************************************* '''

cwd = os.getcwd()
coex_data_list = []

for i,Val0 in enumerate(FFSweepList_0): 
    for j, Val1 in enumerate(FFSweepList_1): 
        RunDirName = str('{}_{}_{}_{}'.format(param_name_0,Val0,param_name_1,Val1))
        RunName = RunDirName
        
        print(RunDirName)
        
        if i == 0 and j == 0:
            coex_guess = coex_init
        elif temp[2] < 0.99*temp[3] or temp[2] > 0.99*temp[2]: # do not seed points that are equal!
            coex_guess = coex_init
        else:
            coex_guess = [temp[2],temp[3]]
            
        print('coex_guess {}'.format(coex_guess))
        
        # Create the Run directory
        c_coex,cost,status,nfev = CreateCGModelDirectory(RunDirName,param_name_0,param_name_1,Val0,Val1,cwd,ForceFieldFileDir,c_low,c_high,coex_guess)
        temp = [Val0, Val1, c_coex[0], c_coex[1], cost, status, nfev]
        coex_data_list.append(temp)
        print('coex_pts {}'.format(c_coex))

header = ('  {}  {}  {}  {}  {}  {}  {}'.format(param_name_0,param_name_1,'C_I','C_II','cost','status','nfev'))
np.savetxt('coex.data',coex_data_list,header=header)        

                   
    
        

