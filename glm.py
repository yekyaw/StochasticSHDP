import numpy as np
import abc
from utils import compute_eta, deriv_helper
from deriv_utils import compute_exp_parts, deriv_C_var_phi, deriv_C_mu

class GLM:
    __metaclass__ = abc.ABCMeta

    def suff_stats(self):
        return np.zeros(self.mu.shape)
    
    @abc.abstractmethod
    def predict(self, var_phi, phi, counts, N):
        return

    @abc.abstractmethod
    def likelihood(self, var_phi, phi, counts, N, y):
        return

    @abc.abstractmethod
    def dphi(self, phi, i, var_phi, counts, N, y, xnorm=None):
        return

    @abc.abstractmethod
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        return
    
    @abc.abstractmethod
    def dmu(self, var_phi, phi, counts, N, y):
        return

    def dsigma(self, var_phi, phi, counts, N, y):
        return 0.

    def update(self, dmu, dsigma, rhot):
        self.mu += rhot * dmu
        self.sigma += rhot * dsigma

    def optimal_ordering(self, idx):
        self.mu = self.mu[idx]
    
    def requires_fp(self):
        return False

class Dummy(GLM):
    def __init__(self, T):
        self.mu = np.zeros(T)
        self.sigma = 1.
        
    def predict(self, var_phi, phi, counts, N):
        return 0
    
    def likelihood(self, var_phi, phi, counts, N, y):
        return 0.
    
    def dphi(self, phi, i, var_phi, counts, N, y, xnorm=None):
        return 0.
    
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        return 0.
    
    def dmu(self, var_phi, phi, counts, N, y):
        return 0.
    
class Poisson(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
        self.sigma = 1.
        
    def predict(self, var_phi, phi, counts, N):
        mean = self._expected_log_norm(var_phi, phi, counts, N)
        return np.round(mean)
        
    def _expected_log_norm(self, var_phi, phi, counts, N):
        _, prod = compute_exp_parts(self.mu, var_phi, phi, counts, N)
        return prod

    def _C_minus_n(self, var_phi, phi, counts, N):
        parts, C = compute_exp_parts(self.mu, var_phi, phi, counts, N)
        C_minus_n = C / (parts + 1e-100)
        return C_minus_n
    
    def likelihood(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        likelihood = y * self.mu.dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi, counts, N)
        return likelihood
    
    def dphi(self, phi, n, var_phi, counts, N, y, xnorm=None):
        dphi = deriv_helper(xnorm, y * var_phi.dot(self.mu) * counts[n] / N)
        C_minus_n = self._C_minus_n(var_phi, phi, counts, N)
        coef = C_minus_n[n] * var_phi.dot(np.exp(self.mu * counts[n] / N))
        dphi -= deriv_helper(xnorm, coef)
        return dphi
    
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, y * phi_mean * self.mu)
        C_minus_n = self._C_minus_n(var_phi, phi, counts, N)
        coef = deriv_C_var_phi(C_minus_n, self.mu, phi[:,i], counts, N)
        dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi
    
    def dmu(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        dmu = y * eta
        C_minus_n = self._C_minus_n(var_phi, phi, counts, N)
        dmu -= deriv_C_mu(C_minus_n, self.mu, phi.dot(var_phi), counts, N)
        return dmu
    
class Categorical(GLM):
    def __init__(self, T, C):
        self.C = C
        self.mu = np.random.normal(0., 0.1, (C, T))
        self.sigma = 1.

    def lda_predict(self, gamma):
        return np.argmax(self.mu.dot(gamma))
        
    def predict(self, var_phi, phi, counts, N):
        eta = compute_eta(var_phi, phi, counts, N)
        mean = self.mu.dot(eta)
        return np.argmax(mean)

    def _all_expected_exps_parts(self, var_phi, phi, counts, N):
        all_parts = np.empty((self.C, counts.shape[0]))
        prods = np.empty(self.C)
        for c in range(self.C):
            parts, prod = compute_exp_parts(self.mu[c,:], var_phi, phi, counts, N)
            all_parts[c,:] = parts
            prods[c] = prod
        return all_parts, prods
    
    def _expected_log_norm(self, var_phi, phi, counts, N):
        _, prods = self._all_expected_exps_parts(var_phi, phi, counts, N)
        sum_prods = np.sum(prods)
        return np.log(sum_prods)

    def likelihood(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        likelihood = self.mu[y,:].dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi, counts, N)
        return likelihood
    
    def dphi(self, phi, n, var_phi, counts, N, y, xnorm=None):
        dphi = deriv_helper(xnorm, var_phi.dot(self.mu[y,:]) * counts[n] / N)
        exps_parts, prods = self._all_expected_exps_parts(var_phi, phi, counts, N)
        sum_prods = np.sum(prods)
        for c in range(self.C):
            C_minus_n = prods[c] / exps_parts[c,n]
            mu_exp = np.exp(self.mu[c,:] * counts[n] / N)
            coef = C_minus_n * var_phi.dot(mu_exp) / sum_prods
            dphi -= deriv_helper(xnorm, coef)
        return dphi
    
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, phi_mean * self.mu[y,:])
        exps_parts, prods = self._all_expected_exps_parts(var_phi, phi, counts, N)
        sum_prods = np.sum(prods)
        for c in range(self.C):
            C_minus_n = prods[c] / exps_parts[c,:]
            coef = deriv_C_var_phi(C_minus_n, self.mu[c,:], phi[:,i], counts, N) / sum_prods
            dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi
    
    def dmu(self, var_phi, phi, counts, N, y):
        dmu = np.zeros(self.mu.shape)        
        eta = compute_eta(var_phi, phi, counts, N)
        dmu[y,:] += eta
        exps_parts, prods = self._all_expected_exps_parts(var_phi, phi, counts, N)
        sum_prods = np.sum(prods)
        for c in range(self.C):
            C_minus_n = prods[c] / exps_parts[c,:]
            dmu[c,:] -= deriv_C_mu(C_minus_n, self.mu[c,:], phi.dot(var_phi), counts, N)
            dmu[c,:] /= sum_prods
        return dmu

    def optimal_ordering(self, idx):
        self.mu = self.mu[:,idx]
    
    def requires_fp(self):
        return True    

class Bernoulli(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
        self.sigma = 1.
        
    def predict(self, var_phi, phi, counts, N):
        eta = compute_eta(var_phi, phi, counts, N)
        mean = self.mu.dot(eta)
        if mean > 0:
            return 1
        else:
            return 0
        
    def _expected_log_norm(self, var_phi, phi, counts, N):
        _, prod = compute_exp_parts(self.mu, var_phi, phi, counts, N)
        return np.log(1 + prod)

    def likelihood(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        likelihood = y * self.mu.dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi, counts, N)
        return likelihood
    
    def dphi(self, phi, n, var_phi, counts, N, y, xnorm=None):
        dphi = deriv_helper(xnorm, y * var_phi.dot(self.mu) * counts[n] / N)
        exp_parts, prod = compute_exp_parts(self.mu, var_phi, phi, counts, N)
        denom = 1 + prod
        C_minus_n = prod / exp_parts[n]
        mu_exp = np.exp(self.mu * counts[n] / N)
        coef = C_minus_n * var_phi.dot(mu_exp) / denom
        dphi -= deriv_helper(xnorm, coef)
        return dphi
    
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, y * phi_mean * self.mu)
        exp_parts, prod = compute_exp_parts(self.mu, var_phi, phi, counts, N)
        denom = 1 + prod
        C_minus_n = prod / exp_parts
        coef = deriv_C_var_phi(C_minus_n, self.mu, phi[:,i], counts, N) / denom
        dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi
    
    def dmu(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        dmu = y * eta
        exp_parts, prod = compute_exp_parts(self.mu, var_phi, phi, counts, N)
        denom = 1 + prod
        C_minus_n = prod / exp_parts
        dmu -= deriv_C_mu(C_minus_n, self.mu, phi.dot(var_phi), counts, N) / denom
        return dmu
    
    def requires_fp(self):
        return True    
    
