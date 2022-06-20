import numpy as np
import time, pickle
import logging
logging.basicConfig(level=logging.INFO)
import multiprocessing as mp

from tramp.algos import ExpectationPropagation, StateEvolution, LogProgress, JoinCallback, NoisyInit, ConstantInit, EarlyStopping, CustomInit
from tramp.priors import GaussianPrior
from Custom_Class.Custom_Channels import ComplexMarchenkoPasturChannel
from Custom_Class.Custom_Callbacks import EarlyStopping_Custom
from tramp.variables import (
    SISOVariable as V, SIMOVariable, MISOVariable, SILeafVariable as O, MILeafVariable
)
from tramp.likelihoods import ModulusLikelihood

#For a product of two complex Gaussian matrices
def mse_se_analytical(alpha, gamma = 1., informed = False, verbosity = 0):
    """
    gamma is the ratio of the matrices. 
    We take the 'generative prior' conventions. The total matrix W = W1*W2 has size mxk, with m/n = alpha
    The matrix W1 is mxn and W2 is nxk, with gamma = n/k
    So W1 has an 'alpha' of alpha, W2 has an 'alpha' of gamma.
    The WR for the recovery of x = W2z should be at alpha = 1/(1+gamma)
    """
    mean = 0.
    tol = 1e-9
    min_variance = 1e-9
    damping = 0.
    alpha_WR = 1./(1.+gamma)
    alpha_PR = min(2.,2./gamma)
    if not(informed): #We add a small mean for the uninformed case, to break the symmetry
        mean = 1e-5 
    size = (2,None) #Since we have an analytical channel, size = None
    prior_z = GaussianPrior(size=size,mean=mean)
    output = ModulusLikelihood(y=None, y_name="y")
    model = prior_z @ V(id="z") @ ComplexMarchenkoPasturChannel(alpha=gamma, name="W2") @ V(id="x") @ ComplexMarchenkoPasturChannel(alpha=alpha, name="W1") @ V(id="W1x") @ output #build model
    model = model.to_model()
    se = StateEvolution(model)
    if not(informed):
        callback = EarlyStopping_Custom(tol=tol, min_variance=min_variance, max_increase = 0.5, min_iterations = 400)
        a0 = 0.
    if informed:
        callback = EarlyStopping_Custom(tol=tol, min_variance=min_variance, max_increase = 0.5, min_iterations = 400)
        a0 = 1e4
        
    if verbosity >= 1:
        callback  = JoinCallback([callback,LogProgress(ids = "all", every = 200)])
    a_init = [("z", "bwd", a0),("x","bwd",a0),("W1x","bwd",a0)]
    initializer = CustomInit(a_init=a_init)
    se.iterate(max_iter = 5000, callback=callback, initializer=initializer,damping=damping)
    records_x = se.get_variables_data('x')['x']
    records_z = se.get_variables_data('z')['z']
    return {'z':records_z,'x':records_x}

def run(alpha,gamma):
    print("Starting alpha = ", alpha, " and gamma = ", gamma)
    t0 = time.time()
    output_uninformed = mse_se_analytical(alpha, gamma = gamma, informed = False, verbosity = 1)
    print("Uninformed step took ", time.time() - t0, " seconds, MSE_z = ",output_uninformed['z']['v'], " and MSE_x = ",output_uninformed['x']['v'])
    t0 = time.time()
    output_informed = mse_se_analytical(alpha,  gamma = gamma, informed = True, verbosity = 1)
    print("Informed step took ", time.time() - t0, " seconds, MSE_z = ",output_informed['z']['v'], " and MSE_x = ",output_informed['x']['v'])
    
    filename = "Data/tmp/results_se_complex_product_gaussians_alpha_"+str(alpha)+"_gamma_"+str(gamma)+".pkl"
    outfile = open(filename,'wb')
    pickle.dump({'uninformed':output_uninformed, 'informed':output_informed},outfile)
    outfile.close()
    return {'uninformed':output_uninformed,'informed':output_informed}

gammas = np.array([1.5,1.0])
for gamma in gammas:
    print("This is the analytical SE for complex Gaussian, with gamma = ", gamma)
    alpha_WR = 1./(1.+gamma)
    alpha_PR = min(2.,2./gamma)
    alphas = np.linspace(alpha_PR*0.999,alpha_PR*1.002,50)
    pool = mp.Pool(processes=12) #The mp pool
    results = [pool.apply(run, args=(alpha,gamma,)) for alpha in alphas]

    
    mses_uninformed = np.array([{'z':result['uninformed']['z']['v'], 'x':result['uninformed']['x']['v']} for result in results])
    mses_informed = np.array([{'z':result['informed']['z']['v'], 'x':result['informed']['x']['v']} for result in results])
    filename = "Data/results_se_complex_product_gaussians_gamma_"+str(gamma)+"_fr_transition.pkl"
    outfile = open(filename,'wb')
    output = {'alphas':alphas, 'mses_uninformed':mses_uninformed,'mses_informed':mses_informed}
    pickle.dump(output,outfile)
    outfile.close()
