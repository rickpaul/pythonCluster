from math import ceil
from matplotlib import pyplot as plt
import scipy.cluster.vq as spc
import numpy as np
# from clusterMarkov import markovChainClusterAlgorithm as MCCAlg
from clusterMarkov import MarkovChainClusterAlgorithm as MCCAlg

if __name__ == '__main__':
	useRandSeed = 0
	seedOverride = None
	useGaussian = 1
	n = 20
	display = 1
	startIndex = 0
	numIterations=100
	numClusters=5
	if not useRandSeed:
		seed = (np.random.randint(9999) if seedOverride is None else seedOverride)
		print 'Seed: ' + str(seed)
		np.random.seed(seed=seed)

	if useGaussian:
		gaussianWidth = 1
		x1 = np.random.randn(n,2) * gaussianWidth + (1,1)
		x2 = np.random.randn(n,2) * gaussianWidth + (3,3)
		x3 = np.random.randn(n,2) * gaussianWidth + (1,1)
		x4 = np.random.randn(n,2) * gaussianWidth + (6,6)
		x5 = np.random.randn(n,2) * gaussianWidth + (9,3)
	else:
		x1 = np.random.random((n,2))-.5 + (1,1)
		x2 = np.random.random((n,2))-.5 + (3,3)
		x3 = np.random.random((n,2))-.5 + (2,3)
		x4 = np.random.random((n,2))-.5 + (6,6)
		x5 = np.random.random((n,2))-.5 + (9,3)
	rawData = np.concatenate((x1,x2,x3,x4,x5),axis=0)

	idx = np.reshape(np.arange(0,len(rawData)),(len(rawData),1))
	x = np.hstack((idx,rawData))

	rollAmount = int(ceil(startIndex/n))

	M = MCCAlg(x,0)
	MCCResults = M.performMCCAlgorithm(startIndex, numIterations=numIterations, numClusters=numClusters, subDataRatio = 0.5)
	transitions = np.roll(np.roll(MCCResults['transitionMatrix'],rollAmount,axis=1),rollAmount,axis=0)
	clusters = np.roll(MCCResults['clusterAssignments'],rollAmount,axis=1)

	np.set_printoptions(precision=3)
	print clusters
	print transitions

	if display:
		plt.scatter(rawData[:,0],rawData[:,1],s=0)
		plt.scatter(rawData[startIndex,0],rawData[startIndex,1],c='r')
		labels = ["{0}".format(i) for i in range(len(rawData))]
		for label, x, y in zip(labels, rawData[:, 0], rawData[:, 1]):
			plt.annotate(label,
				xy = (x, y), xytext = (0, 0),
				textcoords = 'offset points', ha = 'center', va = 'center')
		plt.show()
