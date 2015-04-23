import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
	np.set_printoptions(precision=3)
	display = 1

	useRandSeed = 0
	seedOverride = None

	stringLength = 400
	numAssets = 5 

	if not useRandSeed:
		seed = (np.random.randint(9999) if seedOverride is None else seedOverride)
		print 'Seed: ' + str(seed)
		np.random.seed(seed=seed)

	mean = (1+np.random.random((numAssets))/10)**(1/12.0)-1 #assets return between 0% and 10% p.a.
	std = (np.random.random((numAssets,1))/4 + .05)/(12.0**.5)
	correl = np.random.random((numAssets,numAssets))/2+.2
	correl[range(numAssets),range(numAssets)]=1
	upper = np.triu_indices(numAssets,1)
	lower = (upper[1],upper[0])
	correl[upper] = correl[lower]
	cov = (std * std.T)	* correl
	
	#Can do Cholesky decomposition method, or...
	history = np.random.multivariate_normal(mean, cov,size=(stringLength))
	history = history + np.repeat(np.random.multivariate_normal(mean, cov,size=(stringLength/2)),2,0)/2
	history = history + np.repeat(np.random.multivariate_normal(mean, cov,size=(stringLength/4)),4,0)/2

	words = np.diff(history,1,0) > 0

	print words

	history = np.cumsum(history,0)
	if display:
		plt.plot(range(stringLength),history[:,0])
		plt.plot(range(stringLength),history[:,1])
		plt.plot(range(stringLength),history[:,2])
		plt.plot(range(stringLength),history[:,3])
		plt.plot(range(stringLength),history[:,4])
		plt.show()

	# from clusterBrownian import NGramModel

	# NGramModel = NGramModel(trainingDataSet, indexColumn, nGramSize)