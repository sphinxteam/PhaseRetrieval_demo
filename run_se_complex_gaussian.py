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

#For a complex Gaussian matrix
def mse_se_analytical(alpha, informed = False, verbosity = 0):
    tol = 1e-7
    mean = 0.
    damping = 0. 
    if not(informed): #We add a small mean for the uninformed case, to break the symmetry
        mean = 5e-3
    size = (2,None) #Analytical complex channel
    prior_z = GaussianPrior(size=size,mean=mean)
    output = ModulusLikelihood(y=None, y_name="y")
    model = prior_z @ V(id="z") @ ComplexMarchenkoPasturChannel(alpha=alpha, name="A") @ V(id="Az") @ output #build model
    model = model.to_model()
    se = StateEvolution(model)
    
    if not(informed):
        callback = EarlyStopping(tol=tol, min_variance=1e-12, max_increase = 0.5)
        a0 = 0.
    if informed:
        callback = EarlyStopping_Custom(tol=tol, min_variance=1e-12,max_increase = 0.5, min_iterations = 400)
        a0 = 1e3
    
    if verbosity >= 1:
            callback  = JoinCallback([callback,LogProgress(ids = "all", every = 100)])
    a_init = [("z", "bwd", a0),("Az","bwd",a0)]
    initializer = CustomInit(a_init=a_init)
    se.iterate(max_iter = 5000, callback=callback, initializer=initializer,damping=damping)
    records = se.get_variables_data('z')
    return records

def run_alpha(alpha):
    print("Starting alpha = ", alpha)
    t0 = time.time()
    output_uninformed = mse_se_analytical(alpha, informed = False, verbosity = 1)
    print("Uninformed step took ", time.time() - t0, " seconds, MSE = ",output_uninformed['z']['v'])
    t0 = time.time()
    output_informed = mse_se_analytical(alpha, informed = True, verbosity = 1)
    print("Informed step took ", time.time() - t0, " seconds, MSE = ",output_informed['z']['v'])
    
    #Save partial results
    filename = "Data/tmp/results_se_complex_gaussian_alpha_"+str(alpha)+".pkl"
    outfile = open(filename,'wb')
    pickle.dump({'uninformed':output_uninformed, 'informed':output_informed},outfile)
    outfile.close()
    return {'uninformed':output_uninformed,'informed':output_informed}

print("This is the analytical SE for complex Gaussian")
alphas = np.linspace(0.5, 0.99, 8, endpoint = False)
alphas = np.concatenate((alphas, np.linspace(0.99, 1.05, 24, endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(1.05,1.995,40, endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(1.995,2.03,40, endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(2.03,2.5,5)))
pool = mp.Pool(processes=12) #The mp pool
results = [pool.apply(run_alpha, args=(alpha,)) for alpha in alphas]

#Save final results
mses_uninformed = [result['uninformed']['z']['v'] for result in results]
mses_informed = [result['informed']['z']['v'] for result in results]
mses_uninformed = np.array(mses_uninformed)
mses_informed = np.array(mses_informed)
filename = "Data/results_se_complex_gaussian.pkl"
outfile = open(filename,'wb')
output = {'alphas':alphas, 'mses_uninformed':mses_uninformed,'mses_informed':mses_informed}
pickle.dump(output,outfile)
outfile.close()
