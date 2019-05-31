import numpy as np
from scipy.special import softmax


# Input: two 2D floating point numpy arrays of the same shape.
# The inputs will be interpreted as probability mass functions and the KL divergence is returned.
def kld(p, g):
	if p.ndim != 2:
		raise ValueError("Expected P to be 2 dimensional array")
	if g.ndim != 2:
		raise ValueError("Expected G to be 2 dimensional array")
	if p.shape != g.shape:
		raise ValueError('The shape of P: {} must match the shape of G: {}'.format(p.shape, g.shape))
	if np.any(p < 0):
		raise ValueError('P has some negative values')
	if np.any(g < 0):
		raise ValueError('G has some negative values')

	# Normalize P and G using softmax
	p_n = softmax(p)
	g_n = softmax(g)

	p_n = np.nan_to_num(p)
	g_n = np.nan_to_num(g)

	EPS = 1e-16 # small regularization constant for numerical stability
	kl = np.sum(g_n * np.log2( EPS + (g_n / (EPS + p_n) ) ))

	return kl

