import numpy as np
import time
import logging
import pickle
logging.basicConfig(level=logging.INFO)
import multiprocessing as mp
from scipy import linalg as sp_linalg

from tramp.algos import ExpectationPropagation,StateEvolution, LogProgress, JoinCallback, NoisyInit, ConstantInit, EarlyStopping, CustomInit
from tramp.priors import GaussianPrior
from Custom_Class.Custom_Channels import ComplexMarchenkoPasturChannel, UnitaryChannel, UnitaryLinearChannel_SVD
from tramp.channels import ComplexLinearChannel, ModulusChannel
from Custom_Class.Custom_Ensembles import UnitaryEnsemble_Custom
from tramp.experiments import TeacherStudentScenario
from Custom_Class.Custom_Callbacks import EarlyStopping_Custom
from tramp.variables import (
    SISOVariable as V, SIMOVariable, MISOVariable, SILeafVariable as O, MILeafVariable
)
from tramp.likelihoods import ModulusLikelihood
from tramp.algos import ExpectationPropagation, StateEvolution, LogProgress, JoinCallback, NoisyInit, ConstantInit, EarlyStopping, CustomInit

def mse_gvamp(alpha, N = 5000, verbosity = 0, max_iter = 1000, type = "Haar"):
    assert alpha >= 1, "ERROR : For an unitary matrix, we need alpha >= 1 !"
    """
    - Type can be Haar or DFT
    
    - For an uniform column-orthogonal unitary matrix from the Haar measure, we generate a random matrix W from the Haar measure of size MxM, 
    then we take its first N columns. Then we have the final matrix A = U S V^T, with U = A, V = Id, and S a vector with only 1. 
    
    - For a DFT matrix, we have A = U S conj(V)^T, with U a DFT matrix with a random shuffle of its columns, S a pseudo diagonal of 1s and 
    V a diagonal of random phases
    """
    mean = 5e-3#We add a small mean in the distribution used to infer the data, NOT to generate it.
    size = (2, N) #Shape since we have a complex variables, shape 2
    damping = 0.5
    tol = 1e-6
    min_variance = 1e-8
    
    #We generate the data with no mean
    prior_z = GaussianPrior(size=size, mean=mean)
    prior_z_no_mean = GaussianPrior(size=size, mean=0.)
    
    M = int(alpha * N)
    if type == "Haar":
        U = UnitaryEnsemble_Custom(alpha=1).generate(M) # generate a uniform unitary matrix of size MxM
        A = U[:,0:N]
        s = np.ones(N)
        S = np.zeros((M, N))
        S[:N, :N] = np.diag(s)
        VMatrix = np.eye(N)
        
    elif type == "DFT" or type == "DFT_no_mask":
        #We first generate a DFT matrix of size M, and we normalize it by dividing it by sqrt(M)
        U = sp_linalg.dft(M)/np.sqrt(M) 
        #Then we shuffle randomly its columns 
        
        #U is symmetric, we shuffle its rows
        np.random.shuffle(U)
        #And then put it back in the original way
        U = np.transpose(U)
        
        #The first N columns now give the matrix
        DSmatrix = U[:,:N]
        s = np.ones(N)
        S = np.zeros((M, N))
        S[:N, :N] = np.diag(s)
        
        phases = np.ones(N)
        VMatrix = np.eye(N)
        if type == "DFT":
            #Then we multiply on the right by a diagonal of random phases if we have a mask
            phases = np.exp(1j*np.random.uniform(0,2*np.pi,N))
            VMatrix = np.diag(phases)
        A = np.multiply(DSmatrix,np.conj(phases))
        
        #Now we check if it is unitary
        i = np.random.randint(N-1)
        test_diag = np.dot(A[:,i],np.conj(A[:,i]))
        test_offdiag = np.dot(A[:,i],np.conj(A[:,i+1]))
        if not(np.abs(test_offdiag) <= 1e-5 and np.abs(test_diag-1.) <= 1e-5):
            assert False, "ERROR : The DFT Matrix is not unitary"
        else:
            print("The generated DFT matrix is unitary !")
        
    else:
        assert False, "Error : unkwown type."
        
    print("Matrix generated !")
    model_teacher = prior_z_no_mean @ V(id="z") @ UnitaryLinearChannel_SVD(A,U=U,S=S,V=VMatrix, name="A") @ V(id="Az") @ ModulusChannel() @ O(id="y") # build model
    model_student = prior_z @ V(id="z") @ UnitaryLinearChannel_SVD(A,U=U,S=S,V=VMatrix, name="A")  @ V(id="Az") @ ModulusChannel() @ O(id="y") # build model
    teacher = model_teacher.to_model()
    student = model_student.to_model()
    
    callback = EarlyStopping_Custom(tol=tol, min_variance=min_variance, wait_increase = 10, min_iterations = 200)
    if verbosity >= 1:
        callback  = JoinCallback([LogProgress(ids = "all", every = 50), callback])
    scenario = TeacherStudentScenario(teacher = teacher, student = student, x_ids=["z"])
    scenario.setup()
    records = scenario.run_ep(max_iter=max_iter, damping=damping, callback=callback)
    return records

def run_alpha(alpha, max_iter = 1000):
    Nb_averages = 5 #Number of runs on which we average and take an error bar
    print("Starting alpha = ", alpha, " with " , Nb_averages," averages.")
    output_final_converged_only_Haar = {'mean':-1, 'std':-1}
    output_final_all_Haar = {'mean':-1, 'std':-1}
    output_final_converged_only_DFT = {'mean':-1, 'std':-1}
    output_final_all_DFT = {'mean':-1, 'std':-1}
    outputs_Haar, outputs_converged_only_Haar = [], []
    print("Starting the Haar runs")
    for i in range(Nb_averages):
        print("Number ", i+1, " / ", Nb_averages)
        t0 = time.time()
        output = mse_gvamp(alpha=alpha, verbosity = 1, max_iter =max_iter, type = "Haar")
        print("The step took ", time.time() - t0, " seconds, MSE = ",output['z']['v'])
        n_iter = output['n_iter']
        if n_iter >= max_iter - 1:
            print("THE ALGORITHM HAS NOT CONVERGED IN ", max_iter, " iterations.")
        else:
            outputs_converged_only_Haar.append(output['z']['v'])
        outputs_Haar.append(output['z']['v'])
    output_final_all_Haar['mean'] = np.mean(outputs_Haar)
    output_final_all_Haar['std'] = np.std(outputs_Haar)
    output_final_converged_only_Haar['mean'] = np.mean(outputs_converged_only_Haar)
    output_final_converged_only_Haar['std'] = np.std(outputs_converged_only_Haar)
    
    outputs_DFT, outputs_converged_only_DFT = [], []
    print("Starting the DFT runs")
    for i in range(Nb_averages):
        print("Number ", i+1, " / ", Nb_averages)
        t0 = time.time()
        output = mse_gvamp(alpha=alpha, verbosity = 1, max_iter =max_iter, type = "DFT_no_mask")
        print("The step took ", time.time() - t0, " seconds, MSE = ",output['z']['v'])
        n_iter = output['n_iter']
        if n_iter >= max_iter - 1:
            print("THE ALGORITHM HAS NOT CONVERGED IN ", max_iter, " iterations.")
        else:
            outputs_converged_only_DFT.append(output['z']['v'])
        outputs_DFT.append(output['z']['v'])
    output_final_all_DFT['mean'] = np.mean(outputs_DFT)
    output_final_all_DFT['std'] = np.std(outputs_DFT)
    output_final_converged_only_DFT['mean'] = np.mean(outputs_converged_only_DFT)
    output_final_converged_only_DFT['std'] = np.std(outputs_converged_only_DFT)
    filename = "Data/tmp/results_gvamp_complex_unitary_alpha_"+str(alpha)+".pkl"
    outfile = open(filename,'wb')
    
    result_final = {'Haar':{'all':output_final_all_Haar,'converged_only':output_final_converged_only_Haar},'DFT':{'all':output_final_all_DFT,'converged_only':output_final_converged_only_DFT}}
    
    pickle.dump(result_final,outfile)
    outfile.close()
    return result_final
            
            
print("This is the GVAMP for unitary matrices")
max_iter = 1000
alphas = np.linspace(1.1, 1.99, 5 , endpoint = False)
alphas = np.concatenate((alphas, np.linspace(1.99, 2.3, 15, endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(2.3,2.4, 3)))
pool = mp.Pool(processes=12) #The mp pool

results = [pool.apply(run_alpha, args=(alpha,max_iter,)) for alpha in alphas]
#Save final results
mses_mean_all_Haar = np.array([result['Haar']['all']['mean'] for result in results])
mses_std_all_Haar = np.array([result['Haar']['all']['std'] for result in results])
mses_mean_converged_only_Haar = np.array([result['Haar']['converged_only']['mean'] for result in results])
mses_std_converged_only_Haar = np.array([result['Haar']['converged_only']['std'] for result in results])
mses_mean_all_DFT = np.array([result['DFT']['all']['mean'] for result in results])
mses_std_all_DFT = np.array([result['DFT']['all']['std'] for result in results])
mses_mean_converged_only_DFT = np.array([result['DFT']['converged_only']['mean'] for result in results])
mses_std_converged_only_DFT = np.array([result['DFT']['converged_only']['std'] for result in results])
            
filename = "Data/results_gvamp_complex_unitary.pkl"
outfile = open(filename,'wb')
output = {'alphas':alphas, 'Haar':{   'mses_mean_all':mses_mean_all_Haar,'mses_std_all':mses_std_all_Haar,'mses_mean_converged_only':mses_mean_converged_only_Haar,'mses_std_converged_only':mses_std_converged_only_Haar},'DFT':{   'mses_mean_all':mses_mean_all_DFT,'mses_std_all':mses_std_all_DFT,'mses_mean_converged_only':mses_mean_converged_only_DFT,'mses_std_converged_only':mses_std_converged_only_DFT}}
pickle.dump(output,outfile)
outfile.close()
