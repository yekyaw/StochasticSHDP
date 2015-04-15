import numpy as np
from scipy.misc import logsumexp
import abc

def v_deriv_helper(v_i, i, Esticks, mu, var_phi):
    dig_sum = np.sum(v_i, 0)
    deriv_covs = np.zeros(2)
    mu_dot_var_phi = mu.dot(var_phi[i,:])
    deriv_covs[0] = Esticks[i] * mu_dot_var_phi * v_i[1] / (v_i[0] * dig_sum + 1e-100) 
    deriv_covs[1] = -Esticks[i] * mu_dot_var_phi / (dig_sum + 1e-100)
    deriv_covs[0] -= mu.dot(Esticks[i+1:].dot(var_phi[i+1:,:])) / (dig_sum + 1e-100)
    deriv_covs[1] += mu.dot(Esticks[i+1:].dot(var_phi[i+1:,:])) * v_i[0] / \
                     (v_i[1] * dig_sum + 1e-100)
    return deriv_covs

class GLM:
    __metaclass__ = abc.ABCMeta

    def suff_stats(self):
        return np.zeros(self.mu.shape)

    @abc.abstractmethod
    def predict(self, gamma):
        return

    @abc.abstractmethod
    def likelihood(self, Epi_dot_c, y):
        return

    @abc.abstractmethod
    def dv(self, v_i, i, Esticks_2nd, var_phi, y):
        return

    @abc.abstractmethod
    def dvar_phi(self, xnorm, stick, Epi_dot_c, y):
        return

    @abc.abstractmethod
    def dmu(self, Epi_dot_c, y):
        return

class Dummy(GLM):
    def __init__(self, T):
        self.mu = np.zeros(T)
    
    def predict(self, gamma):
        return 0

    def likelihood(self, Epi_dot_c, y):
        return 0.

    def dv(self, v_i, i, Esticks_2nd, var_phi, y):
        return 0.

    def dvar_phi(self, xnorm, stick, Epi_dot_c, y):
        return 0.

    def dmu(self, Epi_dot_c, y):
        return 0.

class Poisson(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
    def predict(self, gamma):
        eta = self.mu.dot(gamma)
        return np.round(np.exp(eta))
    def likelihood(self, Epi_dot_c, y):
        eta = self.mu.dot(Epi_dot_c)
        likelihood = y * eta
        likelihood -= np.exp(eta)
        return likelihood
    def dv(self, v_i, i, Esticks_2nd, var_phi, y):
        dcovs = v_deriv_helper(v_i, i, Esticks_2nd, self.mu, var_phi)
        deriv = y * dcovs
        Epi_dot_c = Esticks_2nd.dot(var_phi)
        eta = self.mu.dot(Epi_dot_c)
        deriv -= dcovs * np.exp(eta)
        return deriv
    def dvar_phi(self, xnorm, stick, Epi_dot_c, y):
        deriv_helper = lambda xnorm, c : c * xnorm - xnorm * c.dot(xnorm)
        deriv = y * deriv_helper(xnorm, stick * self.mu)
        eta = self.mu.dot(Epi_dot_c)
        coef = stick * self.mu * np.exp(eta)
        deriv -= deriv_helper(xnorm, coef)
        return deriv
    def dmu(self, Epi_dot_c, y):
        dmu = np.zeros(self.mu.shape)
        dmu += y * Epi_dot_c
        eta = self.mu.dot(Epi_dot_c)
        dmu -= Epi_dot_c * np.exp(eta)
        return dmu

class Categorical(GLM):
    def __init__(self, T, C):
        self.C = C
        self.mu = np.random.normal(0., 0.1, (C, T))
    def predict(self, gamma):
        return np.argmax(self.mu.dot(gamma))
    def likelihood(self, Epi_dot_c, y):
        likelihood = self.mu[y,:].dot(Epi_dot_c)
        exps = self.mu.dot(Epi_dot_c)
        likelihood -= logsumexp(exps)
        return likelihood
    def dv(self, v_i, i, Esticks_2nd, var_phi, y):
        dcovs_y = v_deriv_helper(v_i, i, Esticks_2nd, self.mu[y,:], var_phi)
        deriv = dcovs_y
        Epi_dot_c = Esticks_2nd.dot(var_phi)
        exps = self.mu.dot(Epi_dot_c)
        exps_norm = np.exp(exps - logsumexp(exps))
        for c in range(0, self.C):
            dcovs_c = v_deriv_helper(v_i, i, Esticks_2nd, self.mu[c,:], var_phi)
            deriv -= dcovs_c * exps_norm[c]
        return deriv
    def dvar_phi(self, xnorm, stick, Epi_dot_c, y):
        deriv_helper = lambda xnorm, c : c * xnorm - xnorm * c.dot(xnorm)
        exps = self.mu.dot(Epi_dot_c)
        exps_norm = np.exp(exps - logsumexp(exps))
        deriv = deriv_helper(xnorm, stick * self.mu[y,:])
        for c in range(0, self.C):
            coef = stick * self.mu[c,:] * exps_norm[c]
            deriv -= deriv_helper(xnorm, coef)
        return deriv
    def dmu(self, Epi_dot_c, y):
        dmu = np.zeros(self.mu.shape)
        exps = self.mu.dot(Epi_dot_c)
        exps_norm = np.exp(exps - logsumexp(exps))
        dmu[y,:] += Epi_dot_c
        for c in range(0, self.C):
            dmu[c,:] -= Epi_dot_c * exps_norm[c]
        return dmu

class Bernoulli(GLM):
    def __init__(self, T):
        self.mu = np.random.normal(0., 0.1, T)
    def predict(self, gamma):
        if self.mu.dot(gamma) > 0:
            return 1
        else:
            return 0
    def likelihood(self, Epi_dot_c, y):
        eta = self.mu.dot(Epi_dot_c)
        likelihood = y * eta
        likelihood -= np.log(1 + np.exp(eta))
        return likelihood
    def dv(self, v_i, i, Esticks_2nd, var_phi, y):
        dcovs = v_deriv_helper(v_i, i, Esticks_2nd, self.mu, var_phi)
        deriv = y * dcovs
        Epi_dot_c = Esticks_2nd.dot(var_phi)
        eta = self.mu.dot(Epi_dot_c)
        eta_exp = np.exp(eta)
        deriv -= dcovs * eta_exp / (1 + eta_exp)
        return deriv
    def dvar_phi(self, xnorm, stick, Epi_dot_c, y):
        deriv_helper = lambda xnorm, c : c * xnorm - xnorm * c.dot(xnorm)
        deriv = y * deriv_helper(xnorm, stick * self.mu)
        eta = self.mu.dot(Epi_dot_c)
        eta_exp = np.exp(eta)
        coef = stick * self.mu * eta_exp / (1 + eta_exp)
        deriv -= deriv_helper(xnorm, coef)
        return deriv
    def dmu(self, Epi_dot_c, y):
        dmu = np.zeros(self.mu.shape)
        dmu += y * Epi_dot_c
        eta = self.mu.dot(Epi_dot_c)
        eta_exp = np.exp(eta)
        dmu -= Epi_dot_c * eta_exp / (1 + eta_exp)
        return dmu
