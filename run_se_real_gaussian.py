import numpy as np
import time
import logging
import pickle
logging.basicConfig(level=logging.INFO)
import multiprocessing as mp
from scipy import linalg

from tramp.algos import StateEvolution, LogProgress, JoinCallback, NoisyInit, ConstantInit, EarlyStopping, CustomInit
from tramp.priors import GaussianPrior
from tramp.channels import MarchenkoPasturChannel
from Custom_Class.Custom_Callbacks import EarlyStopping_Custom
from tramp.variables import (
    SISOVariable as V, SIMOVariable, MISOVariable, SILeafVariable as O, MILeafVariable
)
from tramp.likelihoods import AbsLikelihood

#For a real Gaussian
def mse_se_analytical(alpha, informed = False, verbosity = 0):
    tol = 1e-7
    mean = 0.
    damping = 0.
    if not(informed): #We add a small mean for the uninformed case, to break the symmetry
        mean = 5e-3
    min_variance = 1e-12
    size = None #Since we have an analytical channel, size = None
    prior_z = GaussianPrior(size=size,mean=mean)
    output = AbsLikelihood(y=None, y_name="y")
    model = prior_z @ V(id="z") @ MarchenkoPasturChannel(alpha=alpha, name="A") @ V(id="Az") @ output #build model
    model = model.to_model()
    se = StateEvolution(model)
    
    if not(informed):
        callback = EarlyStopping(tol=tol, min_variance=min_variance, max_increase = 0.5)
        a0 = 0.
    if informed:
        callback = EarlyStopping_Custom(tol=tol, min_variance=min_variance, max_increase = 0.5, min_iterations = 300)
        a0 = 1e7
    
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
    
    filename = "Data/tmp/results_se_gaussian_alpha_"+str(alpha)+".pkl"
    outfile = open(filename,'wb')
    pickle.dump({'uninformed':output_uninformed, 'informed':output_informed},outfile)
    outfile.close()
    
    return {'uninformed':output_uninformed,'informed':output_informed}

print("Analytical SE for real gaussian matrices")
alphas = np.linspace(0.1, 0.48, 5, endpoint = False)
alphas = np.concatenate((alphas,np.linspace(0.48,0.52,20,endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(0.52,0.97,40,endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(0.97,1.15,40, endpoint = False)))
alphas = np.concatenate((alphas,np.linspace(1.15,1.5,2)))
pool = mp.Pool(processes=12) #The mp pool
results = [pool.apply(run_alpha, args=(alpha,)) for alpha in alphas]

#Save final results
mses_uninformed = [result['uninformed']['z']['v'] for result in results]
mses_informed = [result['informed']['z']['v'] for result in results]
mses_uninformed = np.array(mses_uninformed)
mses_informed = np.array(mses_informed)
filename = "Data/results_se_real_gaussian.pkl"
outfile = open(filename,'wb')
output = {'alphas':alphas, 'mses_uninformed':mses_uninformed,'mses_informed':mses_informed}
pickle.dump(output,outfile)
outfile.close()