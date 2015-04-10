import matplotlib.pyplot as plt
import scipy.cluster.vq as spc
import numpy as np
from clusterMarkov import markovChainClusterAlgorithm as MCCAlg


def performMCCAlgorithm(dataSet, specificDataPointIndex, numIterations = 200, numClusters = 4, subDataRatio = 0.5):
	periodsAhead = np.array([1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 36, 60, 120])
	strippedDataSet = dataSet
	dataLength = strippedDataSet.shape[0]
	dataWidth = strippedDataSet.shape[1]
	specificDataPoint = strippedDataSet[specificDataPointIndex,:]

	numPeriods = len(periodsAhead)

	statisticWeightsbyIteration = np.empty(shape=(numIterations, 4),dtype=float)

	# Perform Bootstrapped Clustering
	for i in range(0,numIterations):
		# Perform Bootstrapped Clustering / Chooose Data Subset
		subDataSetIndexes = np.random.choice(range(0,dataLength),size=dataLength*subDataRatio,replace=True) 
		subDataSet = strippedDataSet[subDataSetIndexes,:]
		# Perform Bootstrapped Clustering / Find Data Clusters for Subset of Data
		kMeansClusters = spc.kmeans(subDataSet, numClusters)
		clusterCenters = kMeansClusters[0]
		# Perform Bootstrapped Clustering / Record Clustering Cost for Weighting Scheme
		clusteringCost = kMeansClusters[1]
		statisticWeightsbyIteration[i,0] = clusteringCost
		# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data
		allClusters = spc.vq(strippedDataSet, clusterCenters)
		clusterAssignments = allClusters[0]
		clusterDistortions = allClusters[1]
		display = 1 #TEST
		if display: #TEST
			plt.scatter(dataSet[0:60,0],dataSet[0:60,1],c=clusterAssignments[0:60]) #TEST	
			plt.show()	
		statisticWeightsbyIteration[i,1] = max(clusterDistortions)
		statisticWeightsbyIteration[i,2] = np.mean(clusterDistortions)
		statisticWeightsbyIteration[i,3] = np.std(clusterDistortions)
	return statisticWeightsbyIteration

n = 12

x1 = np.random.random((n,2))-.5 + (1,1)
x2 = np.random.random((n,2))-.5 + (3,3)
x3 = np.random.random((n,2))-.5 + (1,3)
x4 = np.random.random((n,2))-.5 + (6,6)
x5 = np.random.random((n,2))-.5 + (9,3)
x6 = np.random.random((1,2))-.5 + (90,90)

x = np.concatenate((x1,x2,x3,x4,x5),axis=0)
x = np.concatenate((x1,x2,x3,x4,x5,x6),axis=0)
stats = performMCCAlgorithm(x,1, 100, 6, .8)
print stats

print np.linalg.lstsq(stats[:,1:],stats[:,0])
