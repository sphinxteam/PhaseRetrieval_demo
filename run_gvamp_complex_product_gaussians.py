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

def mse_gvamp(alpha, gamma = 1., n = 5000, verbosity = 0, max_iter = 1000):
    """
    gamma is the ratio of the matrices. 
    We take the 'generative prior' conventions. The total matrix W = W1*W2 has size mxk, with m/n = alpha
    The matrix W1 is mxn and W2 is nxk, with gamma = n/k
    So W1 has an 'alpha' of alpha, W2 has an 'alpha' of gamma.
    The WR for the recovery of x = W2z should be at alpha = 1/(1+gamma)
    """
    mean = 5e-3#We add a small mean in the distribution used to infer the data, NOT to generate it.
    damping = 0.5
    tol = 1e-6
    min_variance = 1e-8
    k = int(n/gamma)
    m = int(n*alpha)
    size = (2, k) #Shape since we have a complex variables, shape 2. Careful, z is of size k here, not n
    
    prior_z = GaussianPrior(size=size, mean=mean)
    prior_z_no_mean = GaussianPrior(size=size, mean=0.)
    W1 = ComplexGaussianEnsemble(m,n).generate() # generate sensing matrix W1
    W2 = ComplexGaussianEnsemble(n,k).generate() # generate sensing matrix W2
    model_teacher = prior_z_no_mean @ V(id="z") @ ComplexLinearChannel(W2, name="W2") @ V(id="x") @ ComplexLinearChannel(W1, name="W1") @ V(id="W1x") @ ModulusChannel() @ O(id="y") # build model
    model_student = prior_z @ V(id="z") @ ComplexLinearChannel(W2, name="W2") @ V(id="x") @ ComplexLinearChannel(W1, name="W1") @ V(id="W1x") @ ModulusChannel() @ O(id="y") # build model
    teacher = model_teacher.to_model()
    student = model_student.to_model()
    
    callback = EarlyStopping_Custom(tol=tol, min_variance=min_variance, wait_increase = 10, min_iterations = 200)
    if verbosity >= 1:
        callback  = JoinCallback([LogProgress(ids = "all", every = 100), callback])
    scenario = TeacherStudentScenario(teacher = teacher, student = student, x_ids=["z","x"])
    scenario.setup()
    records = scenario.run_ep(max_iter=max_iter, damping=damping, callback=callback)
    return records

def run(alpha, gamma, alpha_impossible, max_iter = 1000):
    #alpha_impossible : below this alpha, we only take 2 averages
    Nb_averages = 5 #Number of runs on which we average and take an error bar
    if alpha < alpha_impossible:
        Nb_averages = 2
    print("Starting (gamma,alpha) = (", gamma, "," , alpha, ") with " , Nb_averages," averages.")
    output_final_converged_only = {'z': {'mean':-1, 'std':-1}, 'x':{'mean':-1, 'std':-1}}
    output_final_all = {'z': {'mean':-1, 'std':-1}, 'x':{'mean':-1, 'std':-1}}
    outputs, outputs_converged_only = {'z':[], 'x':[]}, {'z':[], 'x':[]}
    for i in range(Nb_averages):
        print("Number ", i+1, " / ", Nb_averages)
        t0 = time.time()
        output = mse_gvamp(alpha=alpha, gamma = gamma, verbosity = 1, max_iter = max_iter) #We use the default n = 5000
        print("The step took ", time.time() - t0, " seconds, MSE on z = ", output['z']['v'], " and MSE on x = ", output['x']['v'])
        n_iter = output['n_iter']
        if n_iter >= max_iter - 1:
            print("THE ALGORITHM HAS NOT CONVERGED IN ", max_iter, " iterations.")
        else:
            outputs_converged_only['z'].append(output['z']['v'])
            outputs_converged_only['x'].append(output['x']['v'])
        outputs['z'].append(output['z']['v'])
        outputs['x'].append(output['x']['v'])
    output_final_all['z']['mean'] = np.mean(outputs['z'])
    output_final_all['z']['std'] = np.std(outputs['z'])
    output_final_all['x']['mean'] = np.mean(outputs['x'])
    output_final_all['x']['std'] = np.std(outputs['x'])

    output_final_converged_only['z']['mean'] = np.mean(outputs_converged_only['z'])
    output_final_converged_only['z']['std'] = np.std(outputs_converged_only['z'])
    output_final_converged_only['x']['mean'] = np.mean(outputs_converged_only['x'])
    output_final_converged_only['x']['std'] = np.std(outputs_converged_only['x'])

    filename = "Data/tmp/results_gvamp_complex_product_gaussians_gamma_"+str(gamma)+"_alpha_"+str(alpha)+".pkl"
    outfile = open(filename,'wb')
    pickle.dump({'all':output_final_all,'converged_only':output_final_converged_only},outfile)
    outfile.close()
    return {'all':output_final_all, 'converged_only':output_final_converged_only}

max_iter = 1000
print("This is the GVAMP for product of complex Gaussians")
gammas = np.array([0.5, 1.0, 1.5])
alphas = np.array([None for gamma in gammas])
#Building the alpha lists for the different gammas based on the state evolution equations
#gamma = 0.5
alphas[0] = np.linspace(0.3,2/3., 2, endpoint=False)
alphas[0] = np.concatenate((alphas[0], np.linspace(2/3., 2.0, 15, endpoint = False)))
alphas[0] = np.concatenate((alphas[0],np.linspace(2.0,2.1, 3 , endpoint = False)))
alphas[0] = np.concatenate((alphas[0],np.linspace(2.1,2.5,3)))
#gamma = 1.0
alphas[1] = np.linspace(0.25,0.5, 2, endpoint=False)
alphas[1] = np.concatenate((alphas[1], np.linspace(0.5, 2.0, 15, endpoint = False)))
alphas[1] = np.concatenate((alphas[1],np.linspace(2.0,2.5, 6)))
#gamma = 1.5
alphas[2] = np.linspace(0.2,0.4,2, endpoint=False)
alphas[2] = np.concatenate((alphas[2], np.linspace(0.4, 1.4, 15, endpoint = False)))
alphas[2] = np.concatenate((alphas[2],np.linspace(1.4, 2.0, 6)))
#alphas_impossible are the values below which MSE = 1
alphas_impossible = np.array([0.66, 0.49, 0.39])

for (i_gamma, gamma) in enumerate(gammas):
    if i_gamma == 1: #FIXME This is to do only gamma = 1.0 !
        pool = mp.Pool(processes=12) #The mp pool
        alphas_gamma = alphas[i_gamma]
        alpha_impossible = alphas_impossible[i_gamma]
        results = [pool.apply(run, args=(alpha, gamma, alpha_impossible, max_iter,)) for alpha in alphas_gamma]

        #Collecting and saving final results
        mses_mean_all_z = np.array([result['all']['z']['mean'] for result in results])
        mses_std_all_z = np.array([result['all']['z']['std'] for result in results])
        mses_mean_all_x = np.array([result['all']['x']['mean'] for result in results])
        mses_std_all_x = np.array([result['all']['x']['std'] for result in results])
        mses_mean_converged_only_z = np.array([result['converged_only']['z']['mean'] for result in results])
        mses_std_converged_only_z = np.array([result['converged_only']['z']['std'] for result in results])
        mses_mean_converged_only_x = np.array([result['converged_only']['x']['mean'] for result in results])
        mses_std_converged_only_x = np.array([result['converged_only']['x']['std'] for result in results])
        filename = "Data/results_gvamp_complex_product_gaussians_gamma"+str(gamma)+".pkl"
        outfile = open(filename,'wb')
        output = {'alphas':alphas_gamma,
                'mses_mean_all_z':mses_mean_all_z,'mses_std_all_z':mses_std_all_z,'mses_mean_all_x':mses_mean_all_x,'mses_std_all_x':mses_std_all_x,
                'mses_mean_converged_only_z':mses_mean_converged_only_z,'mses_std_converged_only_z':mses_std_converged_only_z, 
                'mses_mean_converged_only_x':mses_mean_converged_only_x,'mses_std_converged_only_x':mses_std_converged_only_x}
        pickle.dump(output,outfile)
        outfile.close()
