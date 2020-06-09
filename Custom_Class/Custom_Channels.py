import numpy as np
from tramp.channels.base_channel import Channel
from tramp.utils.misc import complex2array, array2complex
from .Custom_Ensembles import ComplexMarchenkoPasturEnsemble
from .Custom_Ensembles import SumRealMarchenkoPasturEnsemble
from .Custom_Ensembles import UnitaryEnsemble_Custom, OrthogonalEnsemble_Custom
from tramp.channels import AnalyticalLinearChannel, ComplexLinearChannel, LinearChannel

class AnalyticalComplexLinearChannel(Channel):
    def __init__(self, ensemble, name="W"):
        self.name = name
        self.alpha = ensemble.alpha
        self.repr_init()
        self.ensemble = ensemble

    def sample(self, Z):
        "We assume Z[0] = Z.real and Z[1] = Z.imag"
        Z = array2complex(Z)
        N = Z.shape[0]
        F = self.ensemble.generate(N)
        X = F @ Z
        X = complex2array(X)
        assert X.shape == (2, N), "ERROR : Shape of the complex data is incorrect"
        return X

    def math(self):
        return r"$"+self.name+"$"

    def second_moment(self, tau_z):
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
        n_eff = 1 - self.ensemble.eta_transform(gamma)
        return n_eff    
    
    def compute_backward_error(self, az, ax, tau_z):
        if az==0:
            logger.info(f"az=0 in {self} compute_backward_error")
        az = np.maximum(1e-11, az)
        n_eff = self.compute_n_eff(az, ax)
        vz = (1 - n_eff) / az
        return vz

    def compute_forward_error(self, az, ax, tau_z):
        if ax == 0:
            return self.ensemble.mean_spectrum / (self.alpha * az)
        n_eff = self.compute_n_eff(az, ax)
        vx = n_eff / (self.alpha * ax)
        return vx

    def compute_mutual_information(self, az, ax, tau_z):
        gamma = ax / az
        S = self.ensemble.shannon_transform(gamma)
        I = 0.5*np.log(az*tau_z) + 0.5*S
        return I

    def compute_free_energy(self, az, ax, tau_z):
        tau_x = self.second_moment(tau_z)
        I = self.compute_mutual_information(az, ax, tau_z)
        A = 0.5*(az*tau_z + self.alpha*ax*tau_x) - I + 0.5*np.log(2*np.pi*tau_z/np.e)
        return A
    
class ComplexMarchenkoPasturChannel(AnalyticalComplexLinearChannel):
    def __init__(self, alpha, name = "W"):
        ensemble=ComplexMarchenkoPasturEnsemble(alpha = alpha)
        super().__init__(ensemble = ensemble, name = name)

    #Antoine implemented these 3 functions for debugging, it should work without, with just the init.
    def compute_precision(self, vz, vx, tau_z):
        ax = 1/vx - 1/vz
        az = (1 - self.alpha*ax*vx)/vz
        return az, ax

    def _compute_dual_mutual_information(self, vz, vx, tau_z):
        Iz = 0.5*np.log(tau_z/vz) - 0.5
        J = 0.5*self.alpha*(np.log(vz/vx) + vx/vz - 1)
        I_dual = J + Iz
        return I_dual

    def _compute_dual_free_energy(self, mz, mx, tau_z):
        tau_x = self.second_moment(tau_z)
        vz = tau_z - mz
        vx = tau_x - mx
        I_dual = self._compute_dual_mutual_information(vz, vx, tau_z)
        A_dual = I_dual - 0.5*np.log(2*np.pi*tau_z/np.e)
        return A_dual

class UnitaryChannel(AnalyticalComplexLinearChannel):
    def __init__(self, alpha, name = "W"):
        ensemble=UnitaryEnsemble_Custom(alpha = alpha)
        super().__init__(ensemble = ensemble, name = name)
        
class OrthogonalChannel(AnalyticalLinearChannel):
    def __init__(self, alpha, name = "W"):
        ensemble=OrthogonalEnsemble_Custom(alpha = alpha)
        super().__init__(ensemble = ensemble, name = name)

class UnitaryLinearChannel_SVD(ComplexLinearChannel):
    #For unitary matrices, in order to avoid problems with the numerical SVD, 
    #we write a new UnitaryLinearChannel_SVD class, that directly takes as parameters the (U,S,V) decomposition.
    def __init__(self, W, U, S, V, name="W"):
        """
        U : matrix of size mxm
        V : matrix of size nxn
        S : vector of size min(m,n) = n
        """
        self.name = name
        self.Nx = W.shape[0] #So this is M (>=N)
        self.Nz = W.shape[1] #This is N
        self.precompute_svd = True #To be consistent with the parent class
        self.repr_init()
        self.W = W
        self.rank = np.linalg.matrix_rank(W)
        self.alpha = self.Nx / self.Nz
        
        self.U = U
        self.S = S
        self.V = V
        self.spectrum = np.diag(self.S.conj().T @ self.S)
       
        assert self.spectrum.shape == (self.Nz,)
        self.singular = self.spectrum[:self.rank]

class OrthogonalLinearChannel_SVD(LinearChannel):
    #For orthogonal matrices, in order to avoid problems with the numerical SVD, 
    #we write a new OrthogonalLinearChannel_SVD class, that directly takes as parameters the (U,S,V) decomposition.
    def __init__(self, W, U, S, V, name="W"):
        """
        U : orthogonal matrix of size mxm
        V : orthogonal matrix of size nxn
        S : vector of size min(m,n) = n
        """
        self.name = name
        self.Nx = W.shape[0] #So this is M (>=N)
        self.Nz = W.shape[1] #This is N
        self.precompute_svd = True #To be consistent with the parent class
        self.repr_init()
        self.W = W
        self.rank = np.linalg.matrix_rank(W)
        self.alpha = self.Nx / self.Nz
        
        self.U = U
        self.S = S
        self.V = V
        self.spectrum = np.diag(self.S.T @ self.S)
       
        assert self.spectrum.shape == (self.Nz,)
        self.singular = self.spectrum[:self.rank]
        
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