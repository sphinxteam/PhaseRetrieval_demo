import numpy as np
import time, pickle
import logging
logging.basicConfig(level=logging.INFO)
import multiprocessing as mp

from tramp.algos import ExpectationPropagation, StateEvolution, LogProgress, JoinCallback, NoisyInit, ConstantInit, EarlyStopping, CustomInit
from tramp.priors import GaussianPrior
from tramp.experiments import BayesOptimalScenario, TeacherStudentScenario
from tramp.ensembles import ComplexGaussianEnsemble
from tramp.channels import ComplexLinearChannel, ModulusChannel
from tramp.variables import (
    SISOVariable as V, SIMOVariable, MISOVariable, SILeafVariable as O, MILeafVariable
)
from tramp.likelihoods import ModulusLikelihood
from Custom_Class.Custom_Callbacks import EarlyStopping_Custom

def mse_gvamp(alpha, N = 5000, verbosity = 0, max_iter = 1000):
    mean = 5e-3 #We add a small mean in the distribution used to infer the data, NOT to generate it.
    size = (2, N) 
    damping = 0.5
    tol = 1e-6
    min_variance = 1e-8
    
    prior_z = GaussianPrior(size=size, mean=mean)
    prior_z_no_mean = GaussianPrior(size=size, mean=0.)
    M = int(alpha * N)
    A = ComplexGaussianEnsemble(M,N).generate() # generate sensing matrix A
    model_teacher = prior_z_no_mean @ V(id="z") @ ComplexLinearChannel(A, name="A") @ V(id="Az") @ ModulusChannel() @ O(id="y") # build model
    model_student = prior_z @ V(id="z") @ ComplexLinearChannel(A, name="A") @ V(id="Az") @ ModulusChannel() @ O(id="y") # build model
    teacher = model_teacher.to_model()
    student = model_student.to_model()
    
    callback = EarlyStopping_Custom(tol=tol, min_variance=min_variance, wait_increase = 10, min_iterations = 200)
    if verbosity >= 1:
        callback  = JoinCallback([LogProgress(ids = "all", every = 100), callback])
    scenario = TeacherStudentScenario(teacher = teacher, student = student, x_ids=["z"])
    scenario.setup()
    records = scenario.run_ep(max_iter=max_iter, damping=damping, callback=callback)
    return records

def run_alpha(alpha, max_iter = 1000):
    #Do the runs of GVAMP for a fixed value of alpha
    Nb_averages = 5 #Number of runs on which we average and take an error bar
    if alpha < 0.99:
        Nb_averages = 2
    print("Starting alpha = ", alpha, " with " , Nb_averages," averages.")
    output_final_converged_only = {'mean':-1, 'std':-1}
    output_final_all = {'mean':-1, 'std':-1}
    outputs, outputs_converged_only = [], []
    for i in range(Nb_averages):
        print("Number ", i+1, " / ", Nb_averages)
        t0 = time.time()
        output = mse_gvamp(alpha=alpha, verbosity = 1, max_iter =max_iter)
        print("The step took ", time.time() - t0, " seconds, MSE = ",output['z']['v'])
        n_iter = output['n_iter']
        if n_iter >= max_iter - 1:
            print("THE ALGORITHM HAS NOT CONVERGED IN ", max_iter, " iterations.")
        else:
            outputs_converged_only.append(output['z']['v'])
        outputs.append(output['z']['v'])
    output_final_all['mean'] = np.mean(outputs)
    output_final_all['std'] = np.std(outputs)
    output_final_converged_only['mean'] = np.mean(outputs_converged_only)
    output_final_converged_only['std'] = np.std(outputs_converged_only)
    filename = "Data/tmp/results_gvamp_complex_gaussian_alpha_"+str(alpha)+".pkl"
    outfile = open(filename,'wb')
    pickle.dump({'all':output_final_all,'converged_only':output_final_converged_only},outfile)
    outfile.close()
    return {'all':output_final_all, 'converged_only':output_final_converged_only}

max_iter = 1000
print("This is GVAMP for complex Gaussian")

#The lists of alphas
alphas = np.linspace(0.5, 0.99, 2 , endpoint = False)
alphas = np.concatenate((alphas, np.linspace(0.99, 2.0, 15, endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(2.0,2.05, 3 , endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(2.05,2.2,3)))
pool = mp.Pool(processes=12) #The mp pool

results = [pool.apply(run_alpha, args=(alpha,max_iter,)) for alpha in alphas]
#Save final results
mses_mean_all = np.array([result['all']['mean'] for result in results])
mses_std_all = np.array([result['all']['std'] for result in results])
mses_mean_converged_only = np.array([result['converged_only']['mean'] for result in results])
mses_std_converged_only = np.array([result['converged_only']['std'] for result in results])
filename = "Data/results_gvamp_complex_gaussian.pkl"
outfile = open(filename,'wb')
output = {'alphas':alphas,'mses_mean_all':mses_mean_all,'mses_std_all':mses_std_all,'mses_mean_converged_only':mses_mean_converged_only,'mses_std_converged_only':mses_std_converged_only}
pickle.dump(output,outfile)
outfile.close()
