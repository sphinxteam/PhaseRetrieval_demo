import numpy as np
from tramp.ensembles.base_ensemble import Ensemble
from tramp.ensembles import GaussianEnsemble
from scipy.integrate import quad
from scipy import linalg
import random

class ComplexGaussianEnsemble_Normalized(Ensemble):
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.repr_init()
        self.GE = GaussianEnsemble(M, N)

    def generate(self):
        """Generate complex gaussian iid matrix.

        Returns
        -------
        - X : complex array of shape (M, N)
            X.real and X.imag ~ iid N(var = 1/N)
        """
        X = (self.GE.generate() + 1j*self.GE.generate())/np.sqrt(2)
        return X
    
    
#With the convention that E[|z|^2] = 1, the ensemble is the same than in the real case ! 
class ComplexMarchenkoPasturEnsemble(Ensemble):
    def __init__(self, alpha):
        self.alpha = alpha
        self.repr_init()
        # Minimal and maximal eigenvalues (bulk)
        self.z_max = (1 + np.sqrt(alpha))**2
        self.z_min = (1 - np.sqrt(alpha))**2
        self.mean_spectrum = self.measure(lambda z: z)

    def generate(self, N=1000):
        """Generate complex gaussian iid matrix of size N.
        Returns
        -------
        - X : array of shape (M, N)
        """
        M = int(self.alpha * N)
        sigma_x = 1 / np.sqrt(N)
        X = sigma_x * (np.random.randn(M, N) + 1j*np.random.randn(M, N))/np.sqrt(2)
        return X

    def bulk_density(self, z):
        return np.sqrt((z - self.z_min) * (self.z_max - z))/(2*np.pi*z)

    def measure(self, f):
        atomic = max(0, 1 - self.alpha) * f(0)

        def integrand(z):
            return f(z) * self.bulk_density(z)
        bulk = quad(integrand, self.z_min, self.z_max)[0]
        return atomic + bulk

    def compute_F(self, gamma):
        F = (np.sqrt(gamma*self.z_max + 1) - np.sqrt(gamma*self.z_min + 1))**2
        return F

    def eta_transform(self, gamma):
        F = self.compute_F(gamma)
        return 1 - F/(4*gamma)

    def shannon_transform(self, gamma):
        F = self.compute_F(gamma)
        S = (
            np.log(1 + self.alpha * gamma - F/4) +
            self.alpha * np.log(1 + gamma - F/4) - F / (4*gamma)
        )
        return S
    

class UnitaryEnsemble_Custom(Ensemble):
    def __init__(self, alpha):
        self.alpha = alpha
        assert alpha >= 1, "ERROR : For an unitary matrix, we need alpha >= 1 !"
        self.repr_init()
        self.mean_spectrum = self.measure(lambda z: z)

    def generate(self, N=1000):
        """Generate complex uniform unitary matrix.
        Returns
        -------
        - X : array of shape (M, N) with orthonormal columns
        """
        M = int(self.alpha * N)
        gaussian_matrix = (1./np.sqrt(2.))*(np.random.normal(0, 1., (M, M)) + 1j*np.random.normal(0, 1., (M,M)))
        U, R = linalg.qr(gaussian_matrix)
        #Then we multiply on the right by the phases of the diagonal of R to really get the Haar measure
        D = np.diagonal(R)
        Lambda = D / np.abs(D)
        U = np.multiply(U, Lambda)
        #Then we take its n first columns (scaling of TRAMP needs E[A_{mu i}^2] of order 1/N)
        A = U[:, 0:N]
        return A

    def measure(self, f):
        return f(1.)

    def eta_transform(self, gamma):
        return 1./(1+gamma)

    def shannon_transform(self, gamma):
        return np.log(1.+gamma)

class OrthogonalEnsemble_Custom(Ensemble):
    def __init__(self, alpha):
        self.alpha = alpha
        assert alpha >= 1, "ERROR : For an orthogonal matrix, we need alpha >= 1 !"
        self.repr_init()
        self.mean_spectrum = self.measure(lambda z: z)

    def generate(self, N=1000):
        """Generate uniform orthogonal matrix.
        Returns
        -------
        - X : array of shape (M, N) with orthonormal columns
        """
        M = int(self.alpha * N)
        gaussian_matrix = np.random.normal(0, 1., (M, M))
        U, R = linalg.qr(gaussian_matrix)
        #Then we multiply on the right by the signs of the diagonal of R to really get the Haar measure
        D = np.diagonal(R)
        Lambda = D / np.abs(D)
        U = np.multiply(U, Lambda)
        #Then we take its n first columns (scaling of TRAMP needs E[A_{mu i}^2] of order 1/N)
        A = U[:, 0:N]
        return A

    def measure(self, f):
        return f(1.)

    def eta_transform(self, gamma):
        return 1./(1+gamma)

    def shannon_transform(self, gamma):
        return np.log(1.+gamma)