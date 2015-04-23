import numpy as np
import abc
from utils import compute_eta, deriv_helper

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
    def dmu(self, var_phi, phi, counts, y):
        return

    @abc.abstractmethod
    def requires_fp(self):
        return

    def optimal_ordering(self, idx):
        self.mu = self.mu[idx]

class Dummy(GLM):
    def __init__(self, T):
        self.mu = np.zeros(T)
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
    def requires_fp(self):
        return False    
    
class Poisson(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
    def predict(self, var_phi, phi, counts, N):
        mean = self._expected_log_norm(var_phi, phi, counts, N)
        return np.round(mean)
    def _expected_log_norm_parts(self, var_phi, phi, counts, N):
        mu_exp = np.exp(self.mu / N)
        return np.power(phi.dot(var_phi).dot(mu_exp), counts)    
    def _expected_log_norm(self, var_phi, phi, counts, N):
        parts = self._expected_log_norm_parts(var_phi, phi, counts, N)
        return np.prod(parts)
    def likelihood(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        likelihood = y * self.mu.dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi, counts, N)
        return likelihood
    def dphi(self, phi, n, var_phi, counts, N, y, xnorm=None):
        dphi = deriv_helper(xnorm, y * var_phi.dot(self.mu) * counts[n] / N)
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi, counts, N)        
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / (log_norm_parts[n] + 1e-100)
        coef = counts[n] * log_norm_parts_minus_n * var_phi.dot(np.exp(self.mu / N))
        dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, y * phi_mean * self.mu)
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi, counts, N)
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / (log_norm_parts + 1e-100)
        coef = phi[:,i].dot(log_norm_parts_minus_n * counts) * np.exp(self.mu / N)
        dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi
    def dmu(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        dmu = y * eta
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi, counts, N)
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / (log_norm_parts + 1e-100)
        term = (log_norm_parts_minus_n * counts).dot(phi.dot(var_phi)) / N
        dmu -= term * np.exp(self.mu / N)
        return dmu
    def requires_fp(self):
        return False

class Categorical(GLM):
    def __init__(self, T, C):
        self.C = C
        self.mu = np.random.normal(0., 0.1, (C, T))
    def lda_predict(self, gamma):
        mean = self.mu.dot(gamma)
        return np.argmax(mean)
    def predict(self, var_phi, phi, counts, N):
        eta = compute_eta(var_phi, phi, counts, N)
        mean = self.mu.dot(eta)
        return np.argmax(mean)
    def _expected_exps_parts(self, var_phi, phi, counts, N):
        mu_exp = np.exp(self.mu / N)
        parts = mu_exp.dot(phi.dot(var_phi).T)
        parts = np.apply_along_axis(lambda x: np.power(x, counts), 1, parts)
        return parts
    def _expected_log_norm(self, var_phi, phi, counts, N):
        parts = self._expected_exps_parts(var_phi, phi, counts, N)
        prods = np.prod(parts, 1)
        return np.log(np.sum(prods))
    def likelihood(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        likelihood = self.mu[y,:].dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi, counts, N)
        return likelihood
    def dphi(self, phi, n, var_phi, counts, N, y, xnorm=None):
        dphi = deriv_helper(xnorm, var_phi.dot(self.mu[y,:]) * counts[n] / N)
        exps_parts = self._expected_exps_parts(var_phi, phi, counts, N)
        exps = np.prod(exps_parts, 1)
        denom = np.sum(exps)
        for c in range(self.C):
            exp_parts_minus_n = exps[c] / (exps_parts[c,n] + 1e-100)
            mu_exp = np.exp(self.mu[c,:] / N)
            coef = counts[n] * exp_parts_minus_n * var_phi.dot(mu_exp) / (denom + 1e-100)
            dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, phi_mean * self.mu[y,:])
        exps_parts = self._expected_exps_parts(var_phi, phi, counts, N)
        exps = np.prod(exps_parts, 1)
        denom = np.sum(exps)
        for c in range(self.C):
            exp_parts_minus_n = exps[c] / (exps_parts[c,:] + 1e-100)
            mu_exp = np.exp(self.mu[c,:] / N)
            coef = mu_exp * phi[:,i].dot(exp_parts_minus_n * counts) / (denom + 1e-100)
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
            exp_parts_minus_n = exps[c] / (exps_parts[c,:] + 1e-100)
            term = (exp_parts_minus_n * counts).dot(phi.dot(var_phi)) / N
            dmu[c,:] -= term * np.exp(self.mu[c,:] / N) / denom
        return dmu
    def requires_fp(self):
        return True
    def optimal_ordering(self, idx):
        self.mu = self.mu[:,idx]     

class Bernoulli(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
    def lda_predict(self, gamma):
        mean = self.mu.dot(gamma)
        if mean > 0:
            return 1
        else:
            return 0        
    def predict(self, var_phi, phi, counts, N):
        eta = compute_eta(var_phi, phi, counts, N)
        mean = self.mu.dot(eta)
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
    def dphi(self, phi, n, var_phi, counts, N, y, xnorm=None):
        dphi = deriv_helper(xnorm, y * var_phi.dot(self.mu) * counts[n] / N)
        exp_parts = self._expected_exp_parts(var_phi, phi, counts, N)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / (exp_parts[n] + 1e-100)
        coef = counts[n] * exp_parts_minus_n * var_phi.dot(np.exp(self.mu / N)) / (denom + 1e-100)
        dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, var_phi, i, phi, counts, N, y, xnorm=None):
        phi_mean = phi[:,i].dot(counts) / N
        dvar_phi = deriv_helper(xnorm, y * phi_mean * self.mu)
        exp_parts = self._expected_exp_parts(var_phi, phi, counts, N)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / (exp_parts + 1e-100)
        coef = phi[:,i].dot(exp_parts_minus_n * counts) * np.exp(self.mu / N) / (denom + 1e-100)
        dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi        
    def dmu(self, var_phi, phi, counts, N, y):
        eta = compute_eta(var_phi, phi, counts, N)
        dmu = y * eta
        exp_parts = self._expected_exp_parts(var_phi, phi, counts, N)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / (exp_parts + 1e-100)
        term = (exp_parts_minus_n * counts).dot(phi.dot(var_phi)) / N
        dmu -= term * np.exp(self.mu / N) / denom
        return dmu
    def requires_fp(self):
        return True    
    
