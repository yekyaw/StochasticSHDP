import numpy as np
cimport numpy as np
cimport cython
import ctypes

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "c_utils.h":
    double fill_exp_parts(double *mu, double *phi_dot_var_phi, long *counts,
                          int N, int T, int counts_size, double *parts);
    void fill_deriv_C_var_phi(double *C_minus_n, double *mu, double *phi_k,
                              long *counts, int N, int T, int counts_size, double *deriv);
    void fill_deriv_C_mu(double *C_minus_n, double *mu, double *phi_dot_var_phi,
                              long *counts, int N, int T, int counts_size, double *deriv);
    
@cython.boundscheck(False)
def compute_exp_parts(np.ndarray[DTYPE_t] mu not None, \
                      np.ndarray[DTYPE_t, ndim=2] var_phi not None, \
                      np.ndarray[DTYPE_t, ndim=2] phi not None, \
                      np.ndarray[np.long_t, mode="c"] counts not None, int N):
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] phi_dot_var_phi = phi.dot(var_phi)
    cdef np.ndarray[DTYPE_t, mode="c"] mu_c = np.ascontiguousarray(mu)
    cdef np.ndarray[np.long_t, mode="c"] counts_c = np.ascontiguousarray(counts)      
    cdef int counts_size = counts.shape[0]
    cdef int T = mu.shape[0]
    cdef np.ndarray[DTYPE_t, mode="c"] parts = np.zeros(counts_size)
    prod = fill_exp_parts(&mu_c[0], &phi_dot_var_phi[0,0], <long*>&counts_c[0], N, T, counts_size, &parts[0])
    return (parts, prod)

@cython.boundscheck(False)
def deriv_C_var_phi(np.ndarray[DTYPE_t] C_minus_n not None, \
            np.ndarray[DTYPE_t] mu not None, \
            np.ndarray[DTYPE_t] phi_k not None, \
            np.ndarray[np.long_t] counts not None, int N):
    cdef int T = mu.shape[0]
    cdef int counts_size = counts.shape[0]
    cdef np.ndarray[DTYPE_t, mode="c"] derivs = np.zeros(T)
    cdef np.ndarray[DTYPE_t, mode="c"] C_minus_n_c = np.ascontiguousarray(C_minus_n)    
    cdef np.ndarray[DTYPE_t, mode="c"] phi_k_c = np.ascontiguousarray(phi_k)
    cdef np.ndarray[DTYPE_t, mode="c"] mu_c = np.ascontiguousarray(mu)
    cdef np.ndarray[np.long_t, mode="c"] counts_c = np.ascontiguousarray(counts)
    fill_deriv_C_var_phi(&C_minus_n_c[0], &mu_c[0], &phi_k_c[0], <long*>&counts_c[0], N, T, counts_size, &derivs[0])
    return derivs

@cython.boundscheck(False)
def deriv_C_mu(np.ndarray[DTYPE_t] C_minus_n not None, \
            np.ndarray[DTYPE_t] mu not None, \
            np.ndarray[DTYPE_t, ndim=2] phi_dot_var_phi not None, \
            np.ndarray[np.long_t] counts not None, int N):
    cdef int T = mu.shape[0]
    cdef int counts_size = counts.shape[0]
    cdef np.ndarray[DTYPE_t, mode="c"] derivs = np.zeros(T)
    cdef np.ndarray[DTYPE_t, mode="c"] C_minus_n_c = np.ascontiguousarray(C_minus_n)    
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] phi_dot_var_phi_c = np.ascontiguousarray(phi_dot_var_phi)
    cdef np.ndarray[DTYPE_t, mode="c"] mu_c = np.ascontiguousarray(mu)
    cdef np.ndarray[np.long_t, mode="c"] counts_c = np.ascontiguousarray(counts)    
    fill_deriv_C_mu(&C_minus_n_c[0], &mu_c[0], &phi_dot_var_phi_c[0,0], <long*>&counts_c[0], N, T, counts_size, &derivs[0])
    return derivs
