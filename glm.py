import numpy as np
import abc
from utils import compute_eta, deriv_helper

class GLM:
    __metaclass__ = abc.ABCMeta

    def suff_stats(self):
        return np.zeros(self.mu.shape)
    
    @abc.abstractmethod
    def predict(self, gamma):
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
    def dmu(self, var_phi, phi, counts, y):
        return

    @abc.abstractmethod
    def requires_fp(self):
        return

class Dummy(GLM):
    def __init__(self, T):
        self.mu = np.zeros(T)
    def predict(self, gamma):
        return 0
    def likelihood(self, var_phi, phi, counts, N, y):
        return 0.
    def dphi(self, phi, i, var_phi, counts, N, y, xnorm=None):
        return 0.
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        return 0.
    def dmu(self, var_phi, phi, counts, y):
        return 0.
    def requires_fp(self):
        return False    
    
class Poisson(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
    def predict(self, gamma):
        mean = self.mu.dot(gamma)
        return np.round(np.exp(mean))
    def _expected_log_norm_parts(self, var_phi, phi, counts, N):
        mu_exp = np.exp(self.mu / N)
        return np.power(phi.dot(var_phi).dot(mu_exp), counts)    
    def _expected_log_norm(self, var_phi, phi, counts, N):
        parts = self._expected_log_norm_parts(self.mu, var_phi, phi, counts, N)
        return np.prod(parts)
    def likelihood(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        likelihood = y * self.mu.dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi, counts, N)
        return likelihood
    def dphi(self, phi, i, var_phi, counts, N, y, log_norm_minus_n, xnorm=None):
        dphi = deriv_helper(xnorm, y * var_phi.dot(self.mu) / N)
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi, counts, N)        
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / log_norm_parts[i]
        coef = log_norm_parts_minus_n * var_phi.dot(np.exp(self.mu / N))
        dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, y * phi_mean * self.mu)
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi, counts, N)
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / log_norm_parts
        coef = phi[:,i].dot(log_norm_parts_minus_n) * np.exp(self.mu / N)
        dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi
    def dmu(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        dmu = y * eta
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi, counts, N)
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / log_norm_parts
        term = log_norm_parts_minus_n.dot(phi.dot(var_phi)) / N
        dmu -= term * np.exp(self.mu / N)
        return dmu
    def requires_fp(self):
        return False

class Categorical(GLM):
    def __init__(self, T, C):
        self.C = C
        self.mu = np.random.normal(0., 0.1, (C, T))
    def predict(self, gamma):
        mean = self.mu.dot(gamma)
        return np.argmax(mean)
    def _expected_exps_parts(self, var_phi, phi, counts, N):
        mu_exp = np.exp(self.mu / N)
        parts = mu_exp.dot(phi.dot(var_phi).T)
        parts = np.apply_along_axis(lambda x: np.power(x, counts), 1, parts)
        return parts
    def _expected_log_norm(self, var_phi, phi, counts, N):
        parts = self._expected_exps_parts(var_phi, phi, counts, N)
        return np.log(np.sum(parts))
    def likelihood(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        likelihood = self.mu[y,:].dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi, counts, N)
        return likelihood
    def dphi(self, phi, i, var_phi, counts, N, y, xnorm=None):
        dphi = deriv_helper(xnorm, var_phi.dot(self.mu[y,:]) / N)
        exps_parts = self._expected_exps_parts(var_phi, phi, counts, N)
        exps = np.prod(exps_parts, 1)
        denom = np.sum(exps)
        for c in range(self.C):
            exp_parts_minus_n = exps[c] / exps_parts[c,i]
            coef = exp_parts_minus_n * var_phi.dot(np.exp(self.mu[c,:] / N)) / denom
            dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, phi_mean * self.mu[y,:])
        exps_parts = self._expected_exps_parts(var_phi, phi, counts, N)
        exps = np.prod(exps_parts, 1)
        denom = np.sum(exps)
        for c in range(self.C):
            exp_parts_minus_n = exps[c] / exps_parts[c,:]
            coef = np.exp(self.mu[c,:] / N).dot(phi[:,i].dot(exp_parts_minus_n)) / denom
            dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi        
    def dmu(self, var_phi, phi, counts, N, y):
        dmu = np.zeros(self.mu.shape)        
        eta = compute_eta(var_phi, phi, counts, N)
        dmu[y,:] += eta
        exps_parts = self._expected_exps_parts(var_phi, phi, counts, N)
        exps = np.prod(exps_parts, 1)
        denom = np.sum(exps)
        for c in range(self.C):
            exp_parts_minus_n = exps[c] / exps_parts[c,:]
            term = exp_parts_minus_n.dot(phi.dot(var_phi)) / N
            dmu[c,:] -= term * np.exp(self.mu[c,:] / N) / denom
        return dmu
    def requires_fp(self):
        return True    

class Bernoulli(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
    def predict(self, gamma):
        mean = self.mu.dot(gamma)
        if mean > 0:
            return 1
        else:
            return 0
    def _expected_exp_parts(self, var_phi, phi, counts, N):
        mu_exp = np.exp(self.mu / N)
        return np.power(phi.dot(var_phi).dot(mu_exp), counts)
    def _expected_log_norm(self, var_phi, phi, counts, N):
        parts = self._expected_exp_parts(var_phi, phi, counts, N)
        return np.log(1 + np.prod(parts))
    def likelihood(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        likelihood = y * self.mu.dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi, counts, N)
        return likelihood
    def dphi(self, phi, i, var_phi, counts, N, y, xnorm=None):
        dphi = deriv_helper(xnorm, y * var_phi.dot(self.mu) / N)
        exp_parts = self._expected_exp_parts(var_phi, phi, counts, N)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / exp_parts[i]
        coef = exp_parts_minus_n * var_phi.dot(np.exp(self.mu / N)) / denom
        dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, y * phi_mean * self.mu)
        exp_parts = self._expected_exp_parts(var_phi, phi, counts, N)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / exp_parts
        coef = phi[:,i].dot(exp_parts_minus_n) * np.exp(self.mu / N) / denom
        dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi        
    def dmu(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        dmu = y * eta
        exp_parts = self._expected_exp_parts(var_phi, phi, counts, N)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / exp_parts
        term = exp_parts_minus_n.dot(phi.dot(var_phi)) / N
        dmu -= term * np.exp(self.mu / N) / denom
        return dmu
    def requires_fp(self):
        return True    
    
