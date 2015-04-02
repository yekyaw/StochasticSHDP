#! /usr/bin/python

''' several useful functions '''
import numpy as np
from scipy.linalg import norm

def log_normalize(v):
    ''' return log(sum(exp(v)))'''

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:,np.newaxis]

    return (v, log_norm)

def log_sum(log_a, log_b):
	''' we know log(a) and log(b), compute log(a+b) '''
	v = 0.0;
	if (log_a < log_b):
		v = log_b+np.log(1 + np.exp(log_a-log_b))
	else:
		v = log_a+np.log(1 + np.exp(log_b-log_a))
	return v


def argmax(x):
	''' find the index of maximum value '''
	n = len(x)
	val_max = x[0]
	idx_max = 0

	for i in range(1, n):
		if x[i]>val_max:
			val_max = x[i]
			idx_max = i		

	return idx_max

def backtracking_line_search(f, df, x, p, df_x = None, f_x = None, args = (),
        alpha = 0.0001, beta = 0.9, eps = 1e-8, Verbose = False):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df_x: gradient at x
    f_x = f(x) (Optional)
    args: optional arguments to f (optional)
    alpha, beta: backtracking parameters
    eps: (Optional) quit if norm of step produced is less than this
    Verbose: (Optional) Print lots of info about progress
    
    Reference: Nocedal and Wright 2/e (2006), p. 37
    
    Usage notes:
    -----------
    Recommended for Newton methods; less appropriate for quasi-Newton or conjugate gradients
    """
 
    if f_x is None:
        f_x = f(x, *args)
    if df_x is None:
        df_x = df(x, *args)
 
    assert df_x.T.shape == p.shape
    assert 0 < alpha < 1, 'Invalid value of alpha in backtracking linesearch'
    assert 0 < beta < 1, 'Invalid value of beta in backtracking linesearch'
 
    derphi = np.dot(df_x, p)
 
    assert derphi.shape == (1, 1) or derphi.shape == ()
    assert derphi < 0, 'Attempted to linesearch uphill'
 
    stp = 1.0
    fc = 0
    len_p = norm(p)
 
 
    #Loop until Armijo condition is satisfied
    while f(x + stp * p, *args) > f_x + alpha * stp * derphi:
        stp *= beta
        fc += 1
        if Verbose: print 'linesearch iteration', fc, ':', stp, f(x + stp * p, *args), f_x + alpha * stp * derphi
        if stp * len_p < eps:
            print 'Step is  too small, stop'
            break
    #if Verbose: print 'linesearch iteration 0 :', stp, f_x, f_x
 
    if Verbose: print 'linesearch done'
    #print fc, 'iterations in linesearch'
    return stp

def gradient_descent(f, g, x, max_iter=20, tol=1e-4, proj=None):
    oldf = 0.
    for i in range(max_iter):
        f_x = f(x)
        if np.abs(f_x - oldf) < tol:
            break
        g_x = g(x)
        step = backtracking_line_search(f, g, x, -g_x, df_x=g_x, f_x=f_x)
        x -= step * g_x
        if proj is not None:
            x = proj(x)
        oldf = f_x
    return x

