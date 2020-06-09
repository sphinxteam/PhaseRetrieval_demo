import numpy as np
from scipy import linalg as sp_linalg
import time, logging, pickle, math
logging.basicConfig(level=logging.INFO)
import multiprocessing as mp

from tramp.algos import ExpectationPropagation,StateEvolution, LogProgress, JoinCallback, NoisyInit, ConstantInit, EarlyStopping, CustomInit
from tramp.priors import GaussianPrior
from Custom_Class.Custom_Channels import OrthogonalChannel, OrthogonalLinearChannel_SVD
from tramp.channels import LinearChannel, AbsChannel
from Custom_Class.Custom_Ensembles import OrthogonalEnsemble_Custom
from tramp.experiments import TeacherStudentScenario
from Custom_Class.Custom_Callbacks import EarlyStopping_Custom
from tramp.variables import (
    SISOVariable as V, SIMOVariable, MISOVariable, SILeafVariable as O, MILeafVariable
)
from tramp.likelihoods import AbsLikelihood

def mse_gvamp(alpha, M = 8192, verbosity = 0, max_iter = 1000, type = "Haar"):
    assert alpha >= 1, "ERROR : For an orthogonal matrix, we need alpha >= 1 !"
    """
    - Type can be Haar or Hadamard (randomly subsampled) or Hadamard_no_mask
    
    - For an uniform column-orthogonal matrix from the Haar measure, we generate a random matrix W from the Haar measure of size MxM, 
    then we take its first N columns. Then we have the final matrix A = U S V^T, with U = A, V = Id, and S a vector with only 1. 
    
    - For a Hadamard matrix, we have A = U S conj(V)^T, with U a Hadamard matrix with a random shuffle of its columns, S a pseudo diagonal of 1s and 
    V a diagonal of random signs.

    - For a Hadamard_no_mask this is the same as Hadamard, except V = Id
    """
    mean = 5e-3
    N = int(M/alpha)
    size = N 
    damping = 0.5
    tol = 1e-6
    min_variance = 1e-8
    
    #We generate the data with no mean
    prior_z = GaussianPrior(size=size, mean=mean)
    prior_z_no_mean = GaussianPrior(size=size, mean=0.)

    t0 = time.time()
    if type == "Haar":
        U = OrthogonalEnsemble_Custom(alpha=1).generate(M) # generate a uniform orthogonal matrix of size MxM
        A = U[:,0:N]
        s = np.ones(N)
        S = np.zeros((M, N))
        S[:N, :N] = np.diag(s)
        VMatrix = np.eye(N)
        
    elif type == "Hadamard":

        assert math.log2(M).is_integer(), "ERROR : For an Hadamard matrix, we need M to be a power of 2" #M must be a power of 2
        #We first generate a Hadamard matrix of size M, and we normalize it by dividing it by sqrt(M)
        U = sp_linalg.hadamard(M) / np.sqrt(M)
        #U is symmetric, we shuffle its rows
        np.random.shuffle(U)
        #And then put it back in the original way (to have orthogonal columns)
        U = np.transpose(U)
        
        #The first N columns now give the matrix
        DSmatrix = U[:,:N]
        s = np.ones(N)
        S = np.zeros((M, N))
        S[:N, :N] = np.diag(s)
        
        signs = np.ones(N)
        VMatrix = np.eye(N)
        A = np.multiply(DSmatrix,signs)
        
        #Now we check if it is orthogonal
        i = np.random.randint(N-1)
        test_diag = np.dot(A[:,i],A[:,i])
        test_offdiag = np.dot(A[:,i],A[:,i+1])
        if not(np.abs(test_offdiag) <= 1e-5 and np.abs(test_diag-1.) <= 1e-5):
            assert False, "ERROR : The Hadamard Matrix is not orthogonal"
        else:
            print("The generated Hadamard matrix is orthogonal !")
        
    else:
        assert False, "Error : unkwown type."
        
    t1 = time.time()
    print("Matrix generated in", t1-t0, "seconds.")
    model_teacher = prior_z_no_mean @ V(id="z") @ OrthogonalLinearChannel_SVD(A,U=U,S=S,V=VMatrix, name="A") @ V(id="Az") @ AbsChannel() @ O(id="y") # build model
    model_student = prior_z @ V(id="z") @ OrthogonalLinearChannel_SVD(A,U=U,S=S,V=VMatrix, name="A")  @ V(id="Az") @ AbsChannel() @ O(id="y") # build model
    teacher = model_teacher.to_model()
    student = model_student.to_model()
    
    callback = EarlyStopping_Custom(tol=tol, min_variance=min_variance, wait_increase = 10, min_iterations = 200)
    if verbosity >= 1:
        callback  = JoinCallback([LogProgress(ids = "all", every = 100), callback])
    scenario = TeacherStudentScenario(teacher = teacher, student = student, x_ids=["z"])
    scenario.setup()
    t0 = time.time()
    print("EP scenario created in", t0-t1, "seconds.")
    records = scenario.run_ep(max_iter=max_iter, damping=damping, callback=callback)
    return records

def run_alpha(alpha, alpha_impossible = 1.49, max_iter = 1000):
    Nb_averages = 5 #Number of runs on which we average and take an error bar
    if alpha < alpha_impossible:
        Nb_averages = 2
    print("Starting alpha = ", alpha, " with " , Nb_averages," averages.")
    output_final_converged_only_Haar = {'mean':-1, 'std':-1}
    output_final_all_Haar = {'mean':-1, 'std':-1}
    output_final_converged_only_Hadamard = {'mean':-1, 'std':-1}
    output_final_all_Hadamard = {'mean':-1, 'std':-1}

    outputs_Haar, outputs_converged_only_Haar = [], []
    print("Starting the Haar runs")
    for i in range(Nb_averages):
        print("Number ", i+1, " / ", Nb_averages)
        t0 = time.time()
        output = mse_gvamp(alpha=alpha, verbosity = 1, max_iter =max_iter, type = "Haar") #We take the default value of N
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
    
    outputs_Hadamard, outputs_converged_only_Hadamard = [], []
    print("Starting the Hadamard runs")
    for i in range(Nb_averages):
        print("Number ", i+1, " / ", Nb_averages)
        t0 = time.time()
        output = mse_gvamp(alpha=alpha, verbosity = 1, max_iter =max_iter, type = "Hadamard")
        print("The step took ", time.time() - t0, " seconds, MSE = ",output['z']['v'])
        n_iter = output['n_iter']
        if n_iter >= max_iter - 1:
            print("THE ALGORITHM HAS NOT CONVERGED IN ", max_iter, " iterations.")
        else:
            outputs_converged_only_Hadamard.append(output['z']['v'])
        outputs_Hadamard.append(output['z']['v'])
    output_final_all_Hadamard['mean'] = np.mean(outputs_Hadamard)
    output_final_all_Hadamard['std'] = np.std(outputs_Hadamard)
    output_final_converged_only_Hadamard['mean'] = np.mean(outputs_converged_only_Hadamard)
    output_final_converged_only_Hadamard['std'] = np.std(outputs_converged_only_Hadamard)
    
    filename = "Data/tmp/results_gvamp_orthogonal_alpha_"+str(alpha)+".pkl"
    outfile = open(filename,'wb')
    
    result_final = {'Haar':{'all':output_final_all_Haar,'converged_only':output_final_converged_only_Haar},'Hadamard':{'all':output_final_all_Hadamard,'converged_only':output_final_converged_only_Hadamard}}
    
    pickle.dump(result_final,outfile)
    outfile.close()
    return result_final
            
            
print("This is the GVAMP for orthogonal matrices (Haar matrices and randomly subsampled Hadamard)")
max_iter = 1500
alphas = np.linspace(1.49, 1.6, 15, endpoint = False)
alphas = np.concatenate((alphas,np.linspace(1.6,1.8, 3)))
alphas = np.concatenate((alphas,np.linspace(1.1, 1.49, 5 , endpoint = False))) #I put then last to see if it will be useful to run them again
pool = mp.Pool(processes=12) #The mp pool
alpha_impossible = 1.49

results = [pool.apply(run_alpha, args=(alpha,alpha_impossible, max_iter,)) for alpha in alphas]
#Save final results
mses_mean_all_Haar = np.array([result['Haar']['all']['mean'] for result in results])
mses_std_all_Haar = np.array([result['Haar']['all']['std'] for result in results])
mses_mean_converged_only_Haar = np.array([result['Haar']['converged_only']['mean'] for result in results])
mses_std_converged_only_Haar = np.array([result['Haar']['converged_only']['std'] for result in results])
mses_mean_all_Hadamard = np.array([result['Hadamard']['all']['mean'] for result in results])
mses_std_all_Hadamard = np.array([result['Hadamard']['all']['std'] for result in results])
mses_mean_converged_only_Hadamard = np.array([result['Hadamard']['converged_only']['mean'] for result in results])
mses_std_converged_only_Hadamard = np.array([result['Hadamard']['converged_only']['std'] for result in results])
            
filename = "Data/results_gvamp_orthogonal.pkl"
outfile = open(filename,'wb')
output = {'alphas':alphas, 'Haar':{   'mses_mean_all':mses_mean_all_Haar,'mses_std_all':mses_std_all_Haar,'mses_mean_converged_only':mses_mean_converged_only_Haar,'mses_std_converged_only':mses_std_converged_only_Haar},'Hadamard':{   'mses_mean_all':mses_mean_all_Hadamard,'mses_std_all':mses_std_all_Hadamard,'mses_mean_converged_only':mses_mean_converged_only_Hadamard,'mses_std_converged_only':mses_std_converged_only_Hadamard}}
pickle.dump(output,outfile)
outfile.close()
