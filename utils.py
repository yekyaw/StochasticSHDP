#! /usr/bin/python

''' several useful functions '''
import numpy as np
from scipy.misc import logsumexp

def deriv_helper(xnorm, c):
    return c * xnorm - xnorm * c.dot(xnorm)

def log_normalize(x):
    lognorm = np.tile(logsumexp(x, axis=1), (x.shape[1], 1)).T
    return np.exp(x - lognorm)

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

def compute_eta(var_phi, phi):
    return np.mean(phi.dot(var_phi), 0)
