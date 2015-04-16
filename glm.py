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
    def likelihood(self, var_phi, phi, y):
        return

    @abc.abstractmethod
    def dphi(self, xnorm, phi, i, var_phi, y):
        return    

    @abc.abstractmethod
    def dvar_phi(self, xnorm, var_phi, i, phi, y):
        return
    
    @abc.abstractmethod
    def dmu(self, var_phi, phi, y):
        return

class Dummy(GLM):
    def __init__(self, T):
        self.mu = np.zeros(T)
    def predict(self, gamma):
        return 0
    def likelihood(self, var_phi, phi, y):
        return 0.
    def dphi(self, xnorm, phi, i, var_phi, y):
        return 0.
    def dvar_phi(self, xnorm, var_phi, i, phi, y):
        return 0.
    def dmu(self, var_phi, phi, y):
        return 0.
    
class Poisson(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
    def predict(self, gamma):
        mean = self.mu.dot(gamma)
        return np.round(np.exp(mean))
    def _expected_log_norm_parts(self, var_phi, phi):
        N = phi.shape[0]
        mu_exp = np.exp(self.mu / N)
        return phi.dot(var_phi).dot(mu_exp)
    def _expected_log_norm(self, var_phi, phi):
        parts = self._expected_log_norm_parts(var_phi, phi)
        return np.prod(parts)
    def likelihood(self, var_phi, phi, y):
        eta = compute_eta(var_phi, phi)
        likelihood = y * self.mu.dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi)
        return likelihood
    def dphi(self, xnorm, phi, i, var_phi, y):
        N = phi.shape[0]
        dphi = deriv_helper(xnorm, y * var_phi.dot(self.mu) / N)
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi)
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / (log_norm_parts + 1e-100)
        coef = log_norm_parts_minus_n[i] * var_phi.dot(np.exp(self.mu / N))
        dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, xnorm, var_phi, i, phi, y):
        N = phi.shape[0]
        dvar_phi = deriv_helper(xnorm, y * np.mean(phi[:,i]) * self.mu)
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi)
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / (log_norm_parts + 1e-100)
        coef = phi[:,i].dot(log_norm_parts_minus_n) * np.exp(self.mu / N)
        dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi        
    def dmu(self, var_phi, phi, y):
        N = phi.shape[0]
        eta = compute_eta(var_phi, phi)
        dmu = y * eta
        log_norm_parts = self._expected_log_norm_parts(var_phi, phi)
        log_norm = np.prod(log_norm_parts)
        log_norm_parts_minus_n = log_norm / (log_norm_parts + 1e-100)
        term = log_norm_parts_minus_n.dot(phi.dot(var_phi)) / N
        dmu -= term * np.exp(self.mu / N)
        return dmu

class Categorical(GLM):
    def __init__(self, T, C):
        self.C = C
        self.mu = np.random.normal(0., 0.1, (C, T))
    def predict(self, gamma):
        mean = self.mu.dot(gamma)
        return np.argmax(mean)
    def _expected_exps_parts(self, var_phi, phi):
        N = phi.shape[0]
        mu_exp = np.exp(self.mu / N)
        return mu_exp.dot(phi.dot(var_phi).T)
    def _expected_log_norm(self, var_phi, phi):
        parts = self._expected_exps_parts(var_phi, phi)
        exps = np.prod(parts, 1)
        return np.log(np.sum(exps))
    def likelihood(self, var_phi, phi, y):
        eta = compute_eta(var_phi, phi)
        likelihood = self.mu[y,:].dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi)
        return likelihood
    def dphi(self, xnorm, phi, i, var_phi, y):
        N = phi.shape[0]
        dphi = deriv_helper(xnorm, var_phi.dot(self.mu[y,:]) / N)
        exps_parts = self._expected_exps_parts(var_phi, phi)
        exps = np.prod(exps_parts, 1)
        denom = np.sum(exps)
        for c in range(self.C):
            exp_parts_minus_n = exps[c] / (exps_parts[c,:] + 1e-100)
            coef = exp_parts_minus_n[i] * var_phi.dot(np.exp(self.mu[c,:] / N)) / denom
            dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, xnorm, var_phi, i, phi, y):
        N = phi.shape[0]
        dvar_phi = deriv_helper(xnorm, np.mean(phi[:,i]) * self.mu[y,:])
        exps_parts = self._expected_exps_parts(var_phi, phi)
        exps = np.prod(exps_parts, 1)
        denom = np.sum(exps)
        for c in range(self.C):
            exp_parts_minus_n = exps[c] / (exps_parts[c,:] + 1e-100)
            coef = np.exp(self.mu[c,:] / N).dot(phi[:,i].dot(exp_parts_minus_n)) / denom
            dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi        
    def dmu(self, var_phi, phi, y):
        N = phi.shape[0]
        dmu = np.zeros(self.mu.shape)        
        eta = compute_eta(var_phi, phi)
        dmu[y,:] += eta
        exps_parts = self._expected_exps_parts(var_phi, phi)
        exps = np.prod(exps_parts, 1)
        denom = np.sum(exps)
        for c in range(self.C):
            exp_parts_minus_n = exps[c] / (exps_parts[c,:] + 1e-100)
            term = exp_parts_minus_n.dot(phi.dot(var_phi)) / N
            dmu[c,:] -= term * np.exp(self.mu[c,:] / N) / denom
        return dmu    

class Bernoulli(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
    def predict(self, gamma):
        mean = self.mu.dot(gamma)
        if mean > 0:
            return 1
        else:
            return 0
    def _expected_exp_parts(self, var_phi, phi):
        N = phi.shape[0]
        mu_exp = np.exp(self.mu / N)
        return phi.dot(var_phi).dot(mu_exp)
    def _expected_log_norm(self, var_phi, phi):
        parts = self._expected_exp_parts(var_phi, phi)
        return np.log(1 + np.prod(parts))
    def likelihood(self, var_phi, phi, y):
        eta = compute_eta(var_phi, phi)
        likelihood = y * self.mu.dot(eta)
        likelihood -= self._expected_log_norm(var_phi, phi)
        return likelihood
    def dphi(self, xnorm, phi, i, var_phi, y):
        N = phi.shape[0]
        dphi = deriv_helper(xnorm, y * var_phi.dot(self.mu) / N)
        exp_parts = self._expected_exp_parts(var_phi, phi)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / (exp_parts + 1e-100)
        coef = exp_parts_minus_n[i] * var_phi.dot(np.exp(self.mu / N)) / denom
        dphi -= deriv_helper(xnorm, coef)
        return dphi
    def dvar_phi(self, xnorm, var_phi, i, phi, y):
        N = phi.shape[0]
        dvar_phi = deriv_helper(xnorm, y * np.mean(phi[:,i]) * self.mu)
        exp_parts = self._expected_exp_parts(var_phi, phi)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / (exp_parts + 1e-100)
        coef = phi[:,i].dot(exp_parts_minus_n) * np.exp(self.mu / N) / denom
        dvar_phi -= deriv_helper(xnorm, coef)
        return dvar_phi        
    def dmu(self, var_phi, phi, y):
        N = phi.shape[0]
        eta = compute_eta(var_phi, phi)
        dmu = y * eta
        exp_parts = self._expected_exp_parts(var_phi, phi)
        exp_prod = np.prod(exp_parts)
        denom = 1 + exp_prod
        exp_parts_minus_n = exp_prod / (exp_parts + 1e-100)
        term = exp_parts_minus_n.dot(phi.dot(var_phi)) / N
        dmu -= term * np.exp(self.mu / N) / denom
        return dmu
    
