import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def centeredPCA(X, r):
	mu = np.mean(X, axis=1).reshape((-1,1))
	X = X - mu

	e,U = np.linalg.eigh(X@X.T)
	sorted_idxes = np.argsort(-e)
	e = e[sorted_idxes]
	U = U[:, sorted_idxes]
	Ur = U[:,:r]
	W = Ur.T@X
	return W





def main():

	if len (sys . argv ) != 3 :
		print (sys. argv [0] , " takes 2 arguments . Not ", len (sys. argv ) -1)
		sys. exit ()

	inputfile = sys. argv [1]
	outputfile = sys. argv [2]


	X = np.genfromtxt(inputfile, delimiter=',')
	X = X.T
	r = 2

	pca = centeredPCA(X, r)
	pca = pca.T

	
	np.savetxt(outputfile, pca, delimiter=',')



main()