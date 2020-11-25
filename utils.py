import numpy as np 
def top_k_labels(prob_matrix, k):
	a = prob_matrix.argsort(axis=1)
	b = np.flip(a, axis=1)
	c = b[:,:k]
	return c

