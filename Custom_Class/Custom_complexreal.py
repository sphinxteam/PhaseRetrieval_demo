#The custom quantities for the complex matrix - real signal case

import numpy as np
from tramp.channels.base_channel import Channel
from tramp.utils.misc import complex2array, array2complex
from .Custom_Ensembles import ComplexMarchenkoPasturEnsemble
from .Custom_Ensembles import SumRealMarchenkoPasturEnsemble
from .Custom_Ensembles import UnitaryEnsemble_Custom, OrthogonalEnsemble_Custom
from tramp.channels import AnalyticalLinearChannel, ComplexLinearChannel, LinearChannel

class ComplexGaussianPrior_ZeroIm(Prior):
    #A complex prior with zero imaginary part and gaussian real part
    def __init__(self, size, mean=0, var=1):
        #size is of the type (2,N)
        self.size = size
        assert size[0] == 2, "ERROR : size must be of the type (2,N)"
        self.mean = mean
        self.var = var
        self.repr_init()
        self.sigma = np.sqrt(var)
        self.a = 1 / var #Same conventions as in the real case
        self.b = mean / var

    def sample(self):
        X = self.mean + self.sigma * np.random.standard_normal(self.size)
        #Now we put 0 on the imaginary part
        X[1] = 0.
        return X

    def math(self):
        return r"$\mathcal{N}$"

    def second_moment(self):
        return self.mean**2 + self.var

    def compute_forward_state_evolution(self, ax):
        ax_new = self.a
        return ax_new

#A new channel class for real variables and complex matrix
class AnalyticalComplexRealLinearChannel(Channel):
    def __init__(self, ensemble, name="W"):
        self.name = name
        self.alpha = ensemble.alpha
        self.repr_init()
        self.ensemble = ensemble

    def sample(self, Z):
        "Here Z is real but F is complex" 
        N = Z.shape[0]
        F = self.ensemble.generate(N) #Complex matrix
        X = F @ Z
        X = complex2array(X)
        assert X.shape == (2, N), "ERROR : Shape of the complex data is incorrect"
        return X

    def math(self):
        return r"$"+self.name+"$"

    def second_moment(self, tau_z):
        #Still valid I think
        tau_x = tau_z * (self.ensemble.mean_spectrum / self.alpha)
        return tau_x

    def compute_n_eff(self, az, ax):
        "Effective number of parameters"
        if ax == 0:
            logger.info(f"ax=0 in {self} compute_n_eff")
            return 0.
        if az / ax == 0:
            logger.info(f"az/ax=0 in {self} compute_n_eff")
            return min(1, self.alpha)
        gamma = ax / az
        n_eff = (1 - self.ensemble.eta_transform(gamma)) 
        return n_eff    
    
    def compute_backward_error(self, az, ax, tau_z):
        if az==0:
            logger.info(f"az=0 in {self} compute_backward_error")
        az = np.maximum(1e-11, az)
        n_eff = self.compute_n_eff(az, ax)
        vz = (1 -  n_eff) / az 
        return vz

    def compute_forward_error(self, az, ax, tau_z):
        if ax == 0:
            return self.ensemble.mean_spectrum / (self.alpha * az)
        n_eff = self.compute_n_eff(az, ax) 
        vx = n_eff / (self.alpha * ax) 
        return vx
    
class ComplexRealMarchenkoPasturChannel(AnalyticalComplexRealLinearChannel):
    def __init__(self, alpha, name = "W"):
        ensemble = SumRealMarchenkoPasturEnsemble(alpha = alpha)
        super().__init__(ensemble = ensemble, name = name)
        
        
class SumRealMarchenkoPasturEnsemble(Ensemble):
    #Sum of two real MP matrices, with variance 1/2. This is equivalent to using alpha->2*alpha, and dividing the eigenvalues by 2
    def __init__(self, alpha):
        self.alpha = alpha
        self.realalpha = 2*alpha #We put 2*alpha
        self.sigma = 1.0 #Variance of the gaussian elements
        self.repr_init()
        # Minimal and maximal eigenvalues (bulk)
        self.z_max = self.sigma*(1 + np.sqrt(self.realalpha))**2
        self.z_min = self.sigma*(1 - np.sqrt(self.realalpha))**2
        self.mean_spectrum = self.measure(lambda z: z)
        
    def bulk_density(self, z):
        return np.sqrt((z - self.z_min) * (self.z_max - z))/(2*np.pi*self.sigma*z)

    def measure(self, f):
        atomic = max(0, 1 - self.realalpha) * f(0)

        def integrand(z):
            return f(z) * self.bulk_density(z)
        bulk = quad(integrand, self.z_min, self.z_max)[0]
        return atomic + bulk

    def compute_F(self, gamma):
        F = (np.sqrt(gamma*self.z_max + 1) - np.sqrt(gamma*self.z_min + 1))**2
        return F

    def eta_transform(self, gamma):
        F = self.compute_F(gamma)
        return 1 - F/(4*gamma*self.sigma)

    def shannon_transform(self, gamma):
        F = self.compute_F(gamma)
        S = (
            np.log(1 + self.realalpha * gamma*self.sigma - F/4) +
            self.realalpha * np.log(1 + gamma*self.sigma - F/4) - F / (4*gamma*self.sigma)
        )
        return S