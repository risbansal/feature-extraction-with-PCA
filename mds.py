import numpy as np 
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def mds(X,alpha):

	
	Y = X.reshape(X.shape[0], 1, X.shape[1])
	
	dist = (np.einsum('ijk, ijk->ij', X-Y, X-Y))
	a = float(alpha)/2
	D = np.power(dist,a)

	
	aij = np.square(D)
	ai = np.sum(aij, axis = 1)
	a = np.sum(ai, axis = 0)


	n = D.shape[0]

	G = np.empty(D.shape, dtype = float)

	for i in range(D.shape[0]):
		for j in range(D.shape[1]):
			G[i][j] = (ai[i]/(2*n)) + (ai[j]/(2*n)) - (a/(2*n*n)) - ((aij[i][j])/2)
	
	val,vec = np.linalg.eigh(G)
	sorted_idxes = np.argsort(-val)
	val = val[sorted_idxes]
	vec = vec[:, sorted_idxes]

	val = val[:2]
	vec = vec[:,:2]
	val2 = np.sqrt(val)



	
	u1 = vec[:,0] * val2[0]
	u2 = vec[:,1] * val2[1]

	ans = np.vstack((u1,u2))
	ans = ans.T
	

	return ans



def main():

	if len (sys . argv ) != 4 :
		print (sys. argv [0] , " takes 3 arguments . Not ", len (sys. argv ) -1)
		sys. exit ()

	inputfile = sys. argv [1]
	outputfile = sys. argv [2]
	alpha = sys.argv[3]


	X = np.genfromtxt(inputfile, delimiter=',')

	mds_out = mds(X,alpha)
	np.savetxt(outputfile,mds_out, delimiter=',')

main()
