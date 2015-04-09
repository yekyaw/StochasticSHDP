"""
online hdp with lazy update
part of code is adapted from Matt's online lda code
"""
import numpy as np
cimport numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from scipy.misc import logsumexp
import random, time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

DTYPE = np.longdouble
ctypedef np.double_t DTYPE_t

meanchangethresh = 0.001
min_adding_noise_point = 10
min_adding_noise_ratio = 1 
mu0 = 0.3
rhot_bound = 0.0
burn_in_samples = 5
num_cores = cpu_count()

def split_chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    size = int(len(l) / n)
    left = l
    while(len(left) > 0):
        chunk = left[:size]
        left = left[size:]
        yield chunk

def log_normalize(x):
    lognorm = np.tile(logsumexp(x, axis=1), (x.shape[1], 1)).T
    return np.exp(x - lognorm)

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(sp.psi(alpha) - sp.psi(np.sum(alpha)))
    return(sp.psi(alpha) - sp.psi(np.sum(alpha, 1))[:, np.newaxis])

def expect_sticks(sticks):
    dig_sum = np.sum(sticks, 0)
    E0_W = sticks[0] / dig_sum
    E1_W = sticks[1] / dig_sum

    n = len(sticks[0]) + 1
    Esticks = np.zeros(n)
    Esticks[-1] = 1.
    Esticks[0:n-1] = E0_W
    Esticks[1:] = Esticks[1:] * np.cumprod(E1_W)
    return Esticks 

def expect_log_sticks(np.ndarray[DTYPE_t, ndim=2] sticks):
    """
    For stick-breaking hdp, this returns the E[log(sticks)] 
    """
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n-1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks

def lda_e_step_half(doc, alpha, Elogbeta, split_ratio):

    n_train = int(doc.length * split_ratio)
   
   # split the document
    words_train = doc.words[:n_train]
    counts_train = doc.counts[:n_train]
    words_test = doc.words[n_train:]
    counts_test = doc.counts[n_train:]
    
    gamma = np.ones(len(alpha))  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 

    expElogbeta = np.exp(Elogbeta)
    expElogbeta_train = expElogbeta[:, words_train]
    phinorm = np.dot(expElogtheta, expElogbeta_train) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    max_iter = 100
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        gamma = alpha + expElogtheta * np.dot(counts/phinorm, expElogbeta_train.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, expElogbeta_train) + 1e-100
        meanchange = np.mean(abs(gamma-lastgamma))
        if (meanchange < meanchangethresh):
            break
    gamma = gamma/np.sum(gamma)
    counts = np.array(counts_test)
    expElogbeta_test = expElogbeta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, expElogbeta_test) + 1e-100))

    return (score, np.sum(counts), gamma)

def lda_e_step_split(doc, alpha, beta, max_iter=100):
    half_len = int(doc.length/2) + 1
    idx_train = [2*i for i in range(half_len) if 2*i < doc.length]
    idx_test = [2*i+1 for i in range(half_len) if 2*i+1 < doc.length]
   
   # split the document
    words_train = [doc.words[i] for i in idx_train]
    counts_train = [doc.counts[i] for i in idx_train]
    words_test = [doc.words[i] for i in idx_test]
    counts_test = [doc.counts[i] for i in idx_test]

    gamma = np.ones(len(alpha))  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    betad = beta[:, words_train]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(counts_train)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        gamma = alpha + expElogtheta * np.dot(counts/phinorm,  betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma-lastgamma))
        if (meanchange < meanchangethresh):
            break

    gamma = gamma/np.sum(gamma)
    counts = np.array(counts_test)
    betad = beta[:, words_test]
    score = np.sum(counts * np.log(np.dot(gamma, betad) + 1e-100))

    return (score, np.sum(counts), gamma)

def lda_e_step(doc, alpha, beta, max_iter=100):
    gamma = np.ones(len(alpha))  
    expElogtheta = np.exp(dirichlet_expectation(gamma)) 
    betad = beta[:, doc.words]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(doc.counts)
    iter = 0
    while iter < max_iter:
        lastgamma = gamma
        iter += 1
        likelihood = 0.0
        gamma = alpha + expElogtheta * np.dot(counts/phinorm,  betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = np.mean(abs(gamma-lastgamma))
        if (meanchange < meanchangethresh):
            break

    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha-gamma) * Elogtheta)
    likelihood += np.sum(sp.gammaln(gamma) - sp.gammaln(alpha))
    likelihood += sp.gammaln(np.sum(alpha)) - sp.gammaln(np.sum(gamma))

    return (likelihood, gamma)

class suff_stats:
    def __init__(self, C, T, Wt, Dt):
        self.m_batchsize = Dt
        self.m_var_sticks_ss = np.zeros(T) 
        self.m_var_beta_ss = np.zeros((T, Wt))
        self.m_dmu_ss = [np.zeros((n, T)) for n in C]
    
    def set_zero(self):
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)
        [dmu.fill(0.0) for dmu in self.m_dmu_ss]

class online_hdp:
    ''' hdp model using stick breaking'''
    def __init__(self, C, T, K, D, W, eta, alpha, gamma, kappa, tau, scale=1.0, \
                 adding_noise=False, penalty_lambda=1., l1_ratio=0.6):
        """
        this follows the convention of the HDP paper
        gamma: first level concentration
        alpha: second level concentration
        eta: the topic Dirichlet
        T: top level truncation level
        K: second level truncation level
        W: size of vocab
        D: number of documents in the corpus
        kappa: learning rate
        tau: slow down parameter
        """

        self.m_C = C
        self.m_W = W
        self.m_D = D
        self.m_T = T
        self.m_K = K
        self.m_alpha = alpha
        self.m_gamma = gamma

        self.m_var_sticks = np.zeros((2, T - 1))
        self.m_var_sticks[0] = np.random.gamma(100., 1./100., T - 1)
        self.m_var_sticks[1] = range(T - 1, 0, -1)

        self.m_varphi_ss = np.zeros(T)
        self.m_mu = [np.random.normal(0., 0.1, (n, T)) for n in C]
        self.m_lambda = np.random.gamma(100., 1./100., (T, W))
        self.m_eta = eta
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)

        self.m_penalty_lambda = penalty_lambda
        self.m_l1_ratio = l1_ratio

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_scale = scale
        self.m_updatect = 0 
        self.m_status_up_to_date = True
        self.m_adding_noise = adding_noise
        self.m_num_docs_parsed = 0

        # Timestamps and normalizers for lazy updates
        self.m_timestamp = np.zeros(self.m_W, dtype=int)
        self.m_r = [0]
        self.m_lambda_sum = np.sum(self.m_lambda, axis=1)

    def new_init(self, c):
        self.m_lambda = 1.0/self.m_W + 0.01 * np.random.gamma(1.0, 1.0, \
            (self.m_T, self.m_W))
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)

        num_samples = min(c.num_docs, burn_in_samples)
        ids = random.sample(range(c.num_docs), num_samples)
        docs = [c.docs[id] for id in ids]

        unique_words = dict()
        word_list = []
        for doc in docs:
            for w in doc.words:
                if w not in unique_words:
                    unique_words[w] = len(unique_words)
                    word_list.append(w)
        Wt = len(word_list) # length of words in these documents

        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks
        for doc in docs:
            old_lambda = self.m_lambda[:, word_list].copy()
            for iter in range(5):
                sstats = suff_stats(self.m_C, self.m_T, Wt, 1) 
                doc_score = self.doc_e_step(doc, sstats, Elogsticks_1st, \
                            word_list, unique_words, var_converge=0.0001, max_iter=5)

                self.m_lambda[:, word_list] = old_lambda + sstats.m_var_beta_ss / sstats.m_batchsize
                self.m_Elogbeta = dirichlet_expectation(self.m_lambda)

        self.m_lambda_sum = np.sum(self.m_lambda, axis=1)

    def do_e_step_concurrent(self, docs, Wt, word_list, unique_words, \
                             var_converge, num_workers=num_cores):
        chunks = split_chunks(docs, num_workers)
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks
        args = ((chunk, Elogsticks_1st, Wt, word_list, \
                 unique_words, var_converge) for chunk in chunks)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(self.process_chunk, args)
            return results

    def process_chunk(self, arg):
        (chunk, Elogsticks_1st, Wt, word_list, unique_words, var_converge) = arg
        ss = suff_stats(self.m_C, self.m_T, Wt, len(chunk))
        score = 0.0
        count = 0
        for i, doc in enumerate(chunk):
            doc_score = self.doc_e_step(doc, ss, Elogsticks_1st, \
                        word_list, unique_words, var_converge)
            count += doc.total
            score += doc_score
        return (score, ss, count)

    def process_documents(self, docs, var_converge, unseen_ids=[], update=True, opt_o=False):
        # Find the unique words in this mini-batch of documents...
        self.m_num_docs_parsed += len(docs)
        adding_noise = False
        adding_noise_point = min_adding_noise_point

        if self.m_adding_noise:
            if float(adding_noise_point) / len(docs) < min_adding_noise_ratio:
                adding_noise_point = min_adding_noise_ratio * len(docs)

            if self.m_num_docs_parsed % adding_noise_point == 0:
                adding_noise = True

        unique_words = dict()
        word_list = []
        if adding_noise:
          word_list = range(self.m_W)
          for w in word_list:
            unique_words[w] = w
        else:
            for doc in docs:
                for w in doc.words:
                    if w not in unique_words:
                        unique_words[w] = len(unique_words)
                        word_list.append(w)        
        Wt = len(word_list) # length of words in these documents

        # ...and do the lazy updates on the necessary columns of lambda
        rw = np.array([self.m_r[t] for t in self.m_timestamp[word_list]])
        self.m_lambda[:, word_list] *= np.exp(self.m_r[-1] - rw)
        self.m_Elogbeta[:, word_list] = \
            sp.psi(self.m_eta + self.m_lambda[:, word_list]) - \
            sp.psi(self.m_W*self.m_eta + self.m_lambda_sum[:, np.newaxis])

        
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks

        # run variational inference on some new docs
        score = 0.0
        count = 0
        ss = suff_stats(self.m_C, self.m_T, Wt, 0)

        results = self.do_e_step_concurrent(docs, Wt, word_list, \
                                            unique_words, var_converge)
        for i, result in enumerate(results):
            (chunk_score, chunk_ss, chunk_count) = result
            ss.m_batchsize += chunk_ss.m_batchsize
            ss.m_var_sticks_ss += chunk_ss.m_var_sticks_ss
            ss.m_var_beta_ss += chunk_ss.m_var_beta_ss
            ss.m_dmu_ss = [dmu + chunk_dmu for dmu, chunk_dmu in \
                           zip(ss.m_dmu_ss, chunk_ss.m_dmu_ss)]
            score += chunk_score
            count += chunk_count

        if adding_noise:
            # add noise to the ss
            print("adding noise at this stage...")

            ## old noise
            noise = np.random.gamma(1.0, 1.0, ss.m_var_beta_ss.shape)
            noise_sum = np.sum(noise, axis=1)
            ratio = np.sum(ss.m_var_beta_ss, axis=1) / noise_sum
            noise =  noise * ratio[:,np.newaxis]

            ## new noise
            #lambda_sum_tmp = self.m_W * self.m_eta + self.m_lambda_sum
            #scaled_beta = 5*self.m_W * (self.m_lambda + self.m_eta) / (lambda_sum_tmp[:, np.newaxis])
            #noise = np.random.gamma(shape=scaled_beta, scale=1.0)
            #noise_ratio = lambda_sum_tmp / noise_sum
            #noise = (noise * noise_ratio[:, np.newaxis] - self.m_eta) * len(docs)/self.m_D

            mu = mu0 * 1000.0 / (self.m_updatect + 1000)

            ss.m_var_beta_ss = ss.m_var_beta_ss * (1.0-mu) + noise * mu 
       
        if update:
            self.update_lambda(ss, word_list, opt_o)
    
        return (score, count) 

    def optimal_ordering(self):
        """
        ordering the topics
        """
        idx = [i for i in reversed(np.argsort(self.m_lambda_sum))]
        self.m_varphi_ss = self.m_varphi_ss[idx]
        self.m_lambda = self.m_lambda[idx,:]
        self.m_lambda_sum = self.m_lambda_sum[idx]
        self.m_Elogbeta = self.m_Elogbeta[idx,:]
        self.m_mu = [mu[:,idx] for mu in self.m_mu]

    def deriv_covs(self, np.ndarray[DTYPE_t] v_i, int i, \
                   np.ndarray[DTYPE_t] Esticks,
                   np.ndarray[DTYPE_t, ndim=2] var_phi,
                   np.ndarray[DTYPE_t] mu):
        cdef DTYPE_t dig_sum = np.sum(v_i, 0)
        cdef np.ndarray[DTYPE_t] deriv_covs = np.zeros(2)
        cdef DTYPE_t mu_dot_var_phi = mu.dot(var_phi[i,:])
        deriv_covs[0] = Esticks[i] * mu_dot_var_phi * v_i[1] / (v_i[0] * dig_sum + 1e-100) 
        deriv_covs[1] = -Esticks[i] * mu_dot_var_phi / (dig_sum + 1e-100)
        deriv_covs[0] -= mu.dot(Esticks[i+1:].dot(var_phi[i+1:,:])) / (dig_sum + 1e-100)
        deriv_covs[1] += mu.dot(Esticks[i+1:].dot(var_phi[i+1:,:])) * v_i[0] / \
          (v_i[1] * dig_sum + 1e-100)
        return deriv_covs

    def likelihood_v(self, np.ndarray[DTYPE_t] x, \
                     np.ndarray[DTYPE_t, ndim=2] v, \
                     int i, ys, DTYPE_t ys_scale, \
                     np.ndarray[DTYPE_t, ndim=2] var_phi, \
                     DTYPE_t phi_sum_i, DTYPE_t phi_cum_sum_i):
        cdef np.ndarray[DTYPE_t, ndim=2] temp = v.copy()
        temp[:,i] = x
        cdef DTYPE_t xsum = x[0] + x[1]
        cdef DTYPE_t likelihood = (phi_sum_i - x[0] + 1.) * sp.psi(x[0])
        likelihood += (phi_cum_sum_i + self.m_alpha - x[1]) * sp.psi(x[1])
        cdef DTYPE_t term = 1. + phi_sum_i + phi_cum_sum_i + self.m_alpha
        likelihood -= (term - xsum) * sp.psi(xsum)
        likelihood += sp.gammaln(x[0]) + sp.gammaln(x[1]) - sp.gammaln(xsum)
        cdef np.ndarray[DTYPE_t] Esticks = expect_sticks(temp)
        cdef np.ndarray[DTYPE_t] Epi_dot_c = Esticks.dot(var_phi)
        cdef int y
        cdef np.ndarray[DTYPE_t, ndim=2] mu
        for y, mu in zip(ys, self.m_mu):
            likelihood += mu[y,:].dot(Epi_dot_c) * ys_scale
            exps = mu.dot(Epi_dot_c)
            likelihood -= logsumexp(exps) * ys_scale
        return -likelihood

    def compute_dv(self, np.ndarray[DTYPE_t] x, \
                   np.ndarray[DTYPE_t, ndim=2] v, \
                   int i, ys, DTYPE_t ys_scale, \
                   np.ndarray[DTYPE_t, ndim=2] var_phi, \
                   DTYPE_t phi_sum_i, DTYPE_t phi_cum_sum_i):
        cdef np.ndarray[DTYPE_t, ndim=2] temp = v.copy()
        temp[:,i] = x
        cdef np.ndarray[DTYPE_t] dv = np.zeros(2)
        cdef np.ndarray[DTYPE_t] Esticks_2nd = expect_sticks(temp)
        cdef DTYPE_t xsum = x[0] + x[1]
        cdef DTYPE_t term = 1. + phi_sum_i + phi_cum_sum_i + self.m_alpha
        cdef DTYPE_t term2 = (term - xsum) * sp.polygamma(1, xsum)
        dv[0] = sp.polygamma(1, x[0]) * (phi_sum_i - x[0] + 1.) - term2
        dv[1] = sp.polygamma(1, x[1]) * (phi_cum_sum_i + self.m_alpha - x[1]) - term2

        cdef int y
        cdef int C
        cdef np.ndarray[DTYPE_t] dcovs_y
        cdef np.ndarray[DTYPE_t] dcovs_c
        cdef np.ndarray[DTYPE_t, ndim=2] mu
        cdef np.ndarray[DTYPE_t] Epi_dot_c = Esticks_2nd.dot(var_phi)
        cdef np.ndarray[DTYPE_t] exps
        cdef np.ndarray[DTYPE_t] exps_norm
        cdef int c
        for y, mu, C in zip(ys, self.m_mu, self.m_C):
            dcovs_y = self.deriv_covs(x, i, Esticks_2nd, var_phi, mu[y,:])
            dv += dcovs_y * ys_scale
            exps = mu.dot(Epi_dot_c)
            exps_norm = np.exp(exps - logsumexp(exps))
            for c in range(0, C):
                dcovs_c = self.deriv_covs(x, i, Esticks_2nd, var_phi, mu[c,:])
                dv -= dcovs_c * exps_norm[c] * ys_scale
        return -dv
        
    def _optimize_v(self, np.ndarray[DTYPE_t, ndim=2] v, \
                    np.ndarray[DTYPE_t, ndim=2] phi_all, \
                    np.ndarray[DTYPE_t, ndim=2] var_phi, \
                    ys, DTYPE_t ys_scale):
        cdef np.ndarray[DTYPE_t] phi_sum = np.sum(phi_all[:,:self.m_K-1], 0)
        cdef np.ndarray[DTYPE_t] phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
        cdef np.ndarray[DTYPE_t] phi_cum_sum = np.flipud(np.cumsum(phi_cum))

        bounds = [(1e-100, None)] * 2
        cdef int i
        for i in range(self.m_K - 1):
            args = (v, i, ys, ys_scale, var_phi, phi_sum[i], phi_cum_sum[i])
            res = minimize(self.likelihood_v, v[:,i], method='L-BFGS-B', \
                           jac=self.compute_dv, bounds=bounds, args=args)
            if res.success:
                v[:,i] = res.x
        return v
    
    def likelihood_var_phi(self, np.ndarray[DTYPE_t] x, \
                           np.ndarray[DTYPE_t, ndim=2] var_phi, \
                           int i, ys, DTYPE_t ys_scale, \
                           np.ndarray[DTYPE_t] phi_dot_Elogbeta_i, \
                           np.ndarray[DTYPE_t] Elogsticks_1st, \
                           np.ndarray[DTYPE_t] Esticks_2nd):
        cdef np.ndarray[DTYPE_t] xnorm = np.exp(x - logsumexp(x))
        cdef np.ndarray[DTYPE_t, ndim=2] temp = var_phi.copy()
        temp[i,:] = xnorm
        cdef DTYPE_t likelihood = xnorm.dot(phi_dot_Elogbeta_i)
        likelihood += Elogsticks_1st.dot(xnorm) - xnorm.dot(np.log(xnorm + 1e-100))
        cdef np.ndarray[DTYPE_t] Epi_dot_c = Esticks_2nd.dot(temp)
        cdef np.ndarray[DTYPE_t] exps
        cdef np.ndarray[DTYPE_t, ndim=2] mu
        cdef int y
        for y, mu in zip(ys, self.m_mu):
            likelihood += mu[y,:].dot(Epi_dot_c) * ys_scale
            exps = mu.dot(Epi_dot_c)
            likelihood -= logsumexp(exps) * ys_scale
        return -likelihood

    def compute_dvar_phi(self, np.ndarray[DTYPE_t] x, \
                         np.ndarray[DTYPE_t, ndim=2] var_phi, \
                         int i, ys, DTYPE_t ys_scale, \
                         np.ndarray[DTYPE_t] phi_dot_Elogbeta_i, \
                         np.ndarray[DTYPE_t] Elogsticks_1st, \
                         np.ndarray[DTYPE_t] Esticks_2nd):
        deriv_helper = lambda xnorm, c: c * xnorm - xnorm * c.dot(xnorm)
        cdef np.ndarray[DTYPE_t] xnorm = np.exp(x - logsumexp(x))
        cdef np.ndarray[DTYPE_t, ndim=2] temp = var_phi.copy()
        temp[i,:] = xnorm
        cdef np.ndarray[DTYPE_t] dvar_phi = deriv_helper(xnorm, phi_dot_Elogbeta_i)
        dvar_phi += deriv_helper(xnorm, Elogsticks_1st)
        dvar_phi -= deriv_helper(xnorm, np.ones(len(xnorm)))
        dvar_phi -= deriv_helper(xnorm, np.log(xnorm + 1e-100))
        cdef np.ndarray[DTYPE_t] Epi_dot_c
        cdef np.ndarray[DTYPE_t] exps
        cdef np.ndarray[DTYPE_t] exps_norm
        cdef int c
        cdef np.ndarray[DTYPE_t] term
        cdef np.ndarray[DTYPE_t, ndim=2] mu
        cdef int y
        cdef int C
        for y, mu, C in zip(ys, self.m_mu, self.m_C):
            dvar_phi += deriv_helper(xnorm, Esticks_2nd[i] * mu[y,:]) * ys_scale
            Epi_dot_c = Esticks_2nd.dot(temp)
            exps = mu.dot(Epi_dot_c)
            exps_norm = np.exp(exps - logsumexp(exps))
            for c in range(0, C):
                term = Esticks_2nd[i] * mu[c,:] * exps_norm[c]
                dvar_phi -= deriv_helper(xnorm, term) * ys_scale
        return -dvar_phi

    def _optimize_var_phi(self, np.ndarray[DTYPE_t, ndim=2] var_phi, \
                          np.ndarray[DTYPE_t] Elogsticks_1st, \
                          np.ndarray[DTYPE_t, ndim=2] phi, \
                          np.ndarray[DTYPE_t, ndim=2] Elogbeta, \
                          np.ndarray[DTYPE_t, ndim=2] v, \
                          ys, DTYPE_t ys_scale):
        cdef np.ndarray[DTYPE_t] Esticks_2nd = expect_sticks(v)
        cdef np.ndarray[DTYPE_t, ndim=2] phi_dot_Elogbeta = np.dot(phi.T, Elogbeta.T)        
        cdef int i
        for i in range(self.m_K):
            args = (var_phi, i, ys, ys_scale, phi_dot_Elogbeta[i,:], \
                    Elogsticks_1st, Esticks_2nd)
            res = minimize(self.likelihood_var_phi, var_phi[i,:], \
                           jac=self.compute_dvar_phi, \
                           method='L-BFGS-B', args=args)        
            if res.success:
                x = res.x
                var_phi[i,:] = np.exp(x - logsumexp(x))
        return var_phi

    def _deriv_mu(self, np.ndarray[DTYPE_t, ndim=2] mu, int C,
                  np.ndarray[DTYPE_t] Epi_dot_c, int y, DTYPE_t ys_scale):
        cdef np.ndarray[DTYPE_t, ndim=2] dmu = np.zeros((C, self.m_T))
        cdef np.ndarray[DTYPE_t] exps = mu.dot(Epi_dot_c)
        cdef np.ndarray[DTYPE_t] exps_norm = np.exp(exps - logsumexp(exps))
        dmu[y,:] += Epi_dot_c * ys_scale
        cdef int c
        for c in range(0, C):
            dmu[c,:] -= Epi_dot_c * exps_norm[c] * ys_scale
        return dmu

    def doc_e_step(self, doc, ss, \
                   np.ndarray[DTYPE_t] Elogsticks_1st, \
                   word_list, unique_words, var_converge, \
                   max_iter=20):
        """
        e step for a single doc
        """
        batchids = [unique_words[id] for id in doc.words]
        ys = doc.ys
        cdef DTYPE_t ys_scale = 4.

        cdef np.ndarray[DTYPE_t, ndim=2] Elogbeta_doc = self.m_Elogbeta[:, doc.words]
        ## very similar to the hdp equations
        cdef np.ndarray[DTYPE_t, ndim=2] v = np.zeros((2, self.m_K-1))  
        v[0] = np.random.gamma(100., 1./100., self.m_K - 1)
        v[1] = range(self.m_K - 1, 0, -1)
        
        # The following line is of no use.
        cdef np.ndarray[DTYPE_t] Elogsticks_2nd = expect_log_sticks(v)

        # back to the uniform
        cdef np.ndarray[DTYPE_t, ndim=2] var_phi = np.random.dirichlet(np.ones(self.m_T) * 100., self.m_K)
        cdef np.ndarray[DTYPE_t, ndim=2] phi = np.random.dirichlet(np.ones(self.m_K) * 100., len(doc.words))
        cdef np.ndarray[DTYPE_t] Esticks_2nd = expect_sticks(v)
        cdef np.ndarray[DTYPE_t] Epi_dot_c = Esticks_2nd.dot(var_phi)
        cdef np.ndarray[DTYPE_t] old_Epi_dot_c

        cdef int iter = 0
        # not yet support second level optimization yet, to be done in the future
        while iter < max_iter:
            ### update variational parameters
            # var_phi
            # var_phi = np.dot(phi.T, (Elogbeta_doc * doc.counts).T) + Elogsticks_1st
            # var_phi = log_normalize(var_phi)
            var_phi = self._optimize_var_phi(var_phi, Elogsticks_1st, \
                                             phi, Elogbeta_doc * doc.counts, v, \
                                             ys, ys_scale)
                                
            # phi
            phi = np.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd
            phi = log_normalize(phi)

            # v
            phi_all = phi * np.array(doc.counts)[:,np.newaxis]
            # v[0] = 1.0 + np.sum(phi_all[:,:self.m_K-1], 0)
            # phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
            # v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            v = self._optimize_v(v, phi_all, var_phi, ys, ys_scale)
            Elogsticks_2nd = expect_log_sticks(v)

            Esticks_2nd = expect_sticks(v)
            old_Epi_dot_c = Epi_dot_c
            Epi_dot_c = Esticks_2nd.dot(var_phi)

            iter += 1
            if np.mean(abs(Epi_dot_c - old_Epi_dot_c)) < meanchangethresh:
                break

        cdef DTYPE_t likelihood = 0.0
        # compute likelihood
        # var_phi part/ C in john's notation
        likelihood += np.sum((Elogsticks_1st - np.log(var_phi + 1e-100)) * var_phi)

        # v part/ v in john's notation, john's beta is alpha here
        cdef np.ndarray[DTYPE_t] phi_sum = np.sum(phi_all[:,:self.m_K-1], 0)
        cdef np.ndarray[DTYPE_t] phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
        cdef np.ndarray[DTYPE_t] phi_cum_sum = np.flipud(np.cumsum(phi_cum))
        cdef np.ndarray[DTYPE_t] fixed_terms = 1. + phi_sum + phi_cum_sum + self.m_alpha
        likelihood = np.sum((phi_sum - v[0] + 1.) * sp.psi(v[0]))
        likelihood += np.sum((phi_cum_sum + self.m_alpha - v[1]) * sp.psi(v[1]))
        likelihood -= np.sum((fixed_terms - v[0] - v[1]) * sp.psi(v[0] + v[1]))
        likelihood += np.sum(sp.gammaln(v[0]) + sp.gammaln(v[1]) - sp.gammaln(v[0] + v[1]))

        # Z part 
        likelihood += np.sum((Elogsticks_2nd - np.log(phi + 1e-100)) * phi)

        # X part, the data part
        likelihood += np.sum(phi.T * np.dot(var_phi, Elogbeta_doc * doc.counts))

        # Y part
        cdef int y
        cdef np.ndarray[DTYPE_t, ndim=2] mu
        cdef np.ndarray[DTYPE_t] exps
        for y, mu in zip(ys, self.m_mu):
            likelihood += mu[y,:].dot(Epi_dot_c) * ys_scale
            exps = mu.dot(Epi_dot_c)
            likelihood -= logsumexp(exps) * ys_scale

        # update the suff_stat ss 
        # this time it only contains information from one doc
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   
        ss.m_var_beta_ss[:, batchids] += np.dot(var_phi.T, phi.T * doc.counts)
        ss.m_dmu_ss = [dmu + self._deriv_mu(mu, C, Epi_dot_c, y, ys_scale) \
                       for dmu, y, mu, C in zip(ss.m_dmu_ss, ys, self.m_mu, self.m_C)]
        return(likelihood)

    def doc_e_step_infer(self, doc, Elogsticks_1st, var_converge, max_iter=100):
        """
        e step for a single doc
        """

        Elogbeta_doc = self.m_Elogbeta[:, doc.words]
        ## very similar to the hdp equations
        v = np.zeros((2, self.m_K-1))  
        v[0] = np.random.gamma(100., 1./100., self.m_K - 1)
        v[1] = range(self.m_K - 1, 0, -1)
        
        # The following line is of no use.
        Elogsticks_2nd = expect_log_sticks(v)

        # back to the uniform
        var_phi = np.random.dirichlet(np.ones(self.m_T) * 100., self.m_K)
        phi = np.random.dirichlet(np.ones(self.m_K) * 100., len(doc.words))
        
        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        
        iter = 0
        # not yet support second level optimization yet, to be done in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            ### update variational parameters
            # var_phi
            var_phi = np.dot(phi.T, (Elogbeta_doc * doc.counts).T) + Elogsticks_1st
            var_phi = log_normalize(var_phi)
                                
            # phi
            phi = np.dot(var_phi, Elogbeta_doc).T + Elogsticks_2nd
            phi = log_normalize(phi)

            # v
            phi_all = phi * np.array(doc.counts)[:,np.newaxis]
            v[0] = 1.0 + np.sum(phi_all[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - np.log(var_phi + 1e-100)) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v) * (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

            # Z part 
            likelihood += np.sum((Elogsticks_2nd - np.log(phi + 1e-100)) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Elogbeta_doc * doc.counts))

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            iter += 1

        Esticks_2nd = expect_sticks(v)
        Epi_dot_c = Esticks_2nd.dot(var_phi)        
        return(likelihood, Epi_dot_c)


    def update_lambda(self, sstats, word_list, opt_o):         
        self.m_status_up_to_date = False
        if len(word_list) == self.m_W:
          self.m_status_up_to_date = True
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = self.m_scale * pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound: 
            rhot = rhot_bound
        self.m_rhot = rhot

        # Update appropriate columns of lambda based on documents.
        self.m_lambda[:, word_list] = self.m_lambda[:, word_list] * (1-rhot) + \
            rhot * self.m_D * sstats.m_var_beta_ss / sstats.m_batchsize
        self.m_lambda_sum = (1-rhot) * self.m_lambda_sum + \
            rhot * self.m_D * np.sum(sstats.m_var_beta_ss, axis=1) / sstats.m_batchsize

        def grad_mu(mu, dmu):
            noisy_grad = dmu * self.m_D / sstats.m_batchsize
            noisy_grad -= self.m_penalty_lambda * self.m_l1_ratio * np.sign(mu)
            noisy_grad -= self.m_penalty_lambda * (1 - self.m_l1_ratio) * 2 * mu
            return noisy_grad
        self.m_mu = [mu + rhot * grad_mu(mu, dmu) for mu, dmu in zip(self.m_mu, sstats.m_dmu_ss)]

        self.m_updatect += 1
        self.m_timestamp[word_list] = self.m_updatect
        self.m_r.append(self.m_r[-1] + np.log(1-rhot))

        self.m_varphi_ss = (1.0-rhot) * self.m_varphi_ss + rhot * \
               sstats.m_var_sticks_ss * self.m_D / sstats.m_batchsize

        if opt_o:
            self.optimal_ordering();

        ## update top level sticks 
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T-1]  + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma

        print(expect_sticks(self.m_var_sticks))

    def update_expectations(self):
        """
        Since we're doing lazy updates on lambda, at any given moment
        the current state of lambda may not be accurate. This function
        updates all of the elements of lambda and Elogbeta so that if (for
        example) we want to print out the topics we've learned we'll get the
        correct behavior.
        """
        for w in range(self.m_W):
            self.m_lambda[:, w] *= np.exp(self.m_r[-1] - 
                                          self.m_r[self.m_timestamp[w]])
        self.m_Elogbeta = sp.psi(self.m_eta + self.m_lambda) - \
            sp.psi(self.m_W*self.m_eta + self.m_lambda_sum[:, np.newaxis])
        self.m_timestamp[:] = self.m_updatect
        self.m_status_up_to_date = True

    def print_model(self):
        print(expect_sticks(self.m_var_sticks))
        for mu in self.m_mu:
            print(mu)

    def save_topics(self, filename):
        if not self.m_status_up_to_date:
            self.update_expectations()
        f = open(filename, "w") 
        betas = self.m_lambda + self.m_eta
        for beta in betas:
            line = ' '.join([str(x) for x in beta])  
            f.write(line + '\n')
        f.close()

    def hdp_to_lda(self):
        # compute the lda almost equivalent hdp.
        # alpha
        if not self.m_status_up_to_date:
            self.update_expectations()

        sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T-1] = left      
        alpha = alpha * self.m_alpha
        #alpha = alpha * self.m_gamma
        
        # beta
        beta = (self.m_lambda + self.m_eta) / (self.m_W * self.m_eta + \
            self.m_lambda_sum[:, np.newaxis])

        return (alpha, beta)

    def predict(self, gamma):
        preds = [np.argmax(mu.dot(gamma)) for mu in self.m_mu]
        return preds

    def infer_lda(self, docs):
        lda_alpha, lda_beta = self.hdp_to_lda()
        likelihood = 0
        preds = np.zeros((len(docs), len(self.m_C)))
        gammas = np.zeros((len(docs), self.m_T))
        for i, doc in enumerate(docs):
            (doc_score, gamma) = lda_e_step(doc, lda_alpha, lda_beta)
            likelihood += doc_score
            gammas[i] = gamma
            preds[i,:] = self.predict(gamma)
        return (likelihood, preds, gammas)

    def infer_only(self, docs, var_converge):
        # be sure to run update_expectations()
        if not self.m_status_up_to_date:
            self.update_expectations()
        #alpha = alpha * self.m_gamma
        likelihood = 0
        preds = np.zeros((len(docs), len(self.m_C)))
        gammas = np.zeros((len(docs), self.m_T))
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks)
        for i, doc in enumerate(docs):
            (doc_score, gamma) = self.doc_e_step_infer(doc, Elogsticks_1st, var_converge)
            likelihood += doc_score
            gammas[i] = gamma
            preds[i,:] = self.predict(gamma)
        return (likelihood, preds, gammas)
