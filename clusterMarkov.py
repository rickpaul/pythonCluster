#TODO: use sklearn.cluster to use other cluster types
import scipy.cluster.vq as spc
import numpy as np
from math import floor


class MarkovChainClusterAlgorithm:

	def __init__(self,dataSet, indexColumn, dateColumn = None):
		self.dataSet = dataSet
		self.indexColumn = indexColumn
		self.dateColumn = dateColumn

	# Complex Distributions Version (produces weighted distribution for histogram output)
	# Dissatisfied with the weighting scheme...
	def performMCCAlgorithm(self, specificDataPointIndex, numIterations = 200, numClusters = 4, subDataRatio = 0.5):
		#We can also change this based on a verification algorithm (i.e. use test data to test training data)
		strippedDataSet = self.stripDataSet(self.dataSet, self.dateColumn, self.indexColumn)
		dataLength = strippedDataSet.shape[0]
		dataWidth = strippedDataSet.shape[1]

		clusterCounter = np.zeros(shape=(dataLength, numClusters*2),dtype=float) #TEST for ATTEMPT 3
		transitionCounter = np.zeros(shape=(numClusters*2, numClusters*2),dtype=float)
		# clusterCounter = np.zeros(shape=(dataLength, numClusters),dtype=float)
		# transitionCounter = np.zeros(shape=(numClusters, numClusters),dtype=float)

		# Perform Bootstrapped Clustering
		for j in range(0,numIterations):
			# Perform Bootstrapped Clustering / Chooose Data Subset
			subDataSetIndexes = np.random.choice(range(0,dataLength),size=dataLength*subDataRatio,replace=True) 
			subDataSet = strippedDataSet[subDataSetIndexes,:]
			# Perform Bootstrapped Clustering / Find Data Clusters for Subset of Data
			kMeansClusters = spc.kmeans(subDataSet, numClusters)
			clusterCenters = kMeansClusters[0]
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data
			allClusters = spc.vq(strippedDataSet, clusterCenters)
			clusterAssignments = np.array(allClusters[0])
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data / Find Cluster Statistics
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data / Find Cluster Statistics / Reorder Clusters to Stable Order
			#TODO: Make more efficient?
			#TODO: Clean up cluster assignment
			####################
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data / Find Cluster Statistics / Reorder Clusters to Stable Order / Attempt 1
			# orderedClusterAssignments = np.zeros(shape=(dataLength, 1),dtype=int)-1
			# currentClusterAssignment = beginningClusterAssignment = 0
			# for i in range(specificDataPointIndex,dataLength):
			# 	if orderedClusterAssignments[i] >= 0:
			# 		continue
			# 	sameClusters = np.arange(0,dataLength)[clusterAssignments == clusterAssignments[i]]
			# 	orderedClusterAssignments[sameClusters] = currentClusterAssignment
			# 	clusterCounter[sameClusters, currentClusterAssignment] +=1
			# 	currentClusterAssignment = currentClusterAssignment + 1
			# 	if currentClusterAssignment == numClusters:
			# 		break
			# finalAssignedCluster = currentClusterAssignment - 1
			# currentClusterAssignment = numClusters - 1
			# for i in range(specificDataPointIndex,0,-1):
			# 	if currentClusterAssignment == finalAssignedCluster:
			# 		break
			# 	if orderedClusterAssignments[i] >= 0:
			# 		continue
			# 	sameClusters = np.arange(0,dataLength)[clusterAssignments == clusterAssignments[i]]
			# 	orderedClusterAssignments[sameClusters] = currentClusterAssignment
			# 	clusterCounter[sameClusters, currentClusterAssignment] +=1
			# 	currentClusterAssignment = currentClusterAssignment - 1
			####################
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data / Find Cluster Statistics / Reorder Clusters to Stable Order / Attempt 2
			# orderedClusterAssignments = np.zeros(shape=(dataLength, 1),dtype=int)-1
			# currentClusterAssignment = 0
			# for i in range(dataLength):
			# 	if orderedClusterAssignments[i] >= 0:
			# 		continue
			# 	sameClusters = np.arange(0,dataLength)[clusterAssignments == clusterAssignments[i]]
			# 	orderedClusterAssignments[sameClusters] = currentClusterAssignment
			# 	currentClusterAssignment = currentClusterAssignment + 1
			# 	if currentClusterAssignment == numClusters:
			# 		break
			# specificDataPointCluster = orderedClusterAssignments[specificDataPointIndex]
			# orderedClusterAssignments = (orderedClusterAssignments - specificDataPointCluster) % numClusters
			# # Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data / Find Cluster Statistics / Fill Cluster Count Matrix
			# for i in range(dataLength):
			# 	clusterCounter[i, orderedClusterAssignments[i]] +=1
			####################
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data / Find Cluster Statistics / Reorder Clusters to Stable Order / Attempt 2
			orderedClusterAssignments = np.zeros(shape=(dataLength, 1),dtype=int)-1
			currentClusterAssignment = beginningClusterAssignment = 0
			for i in range(specificDataPointIndex,dataLength):
				if orderedClusterAssignments[i] >= 0:
					continue
				sameClusters = np.arange(0,dataLength)[clusterAssignments == clusterAssignments[i]]
				orderedClusterAssignments[sameClusters] = currentClusterAssignment
				clusterCounter[sameClusters, currentClusterAssignment] +=1
				currentClusterAssignment = currentClusterAssignment + 1
				if currentClusterAssignment == numClusters:
					break
			finalAssignedCluster = currentClusterAssignment - 1 #because we incremented it after final assigment, need to decrement it...
			currentClusterAssignment = numClusters*2 - 1
			for i in range(specificDataPointIndex,0,-1):
				if currentClusterAssignment == finalAssignedCluster:
					break
				if orderedClusterAssignments[i] >= 0:
					continue
				sameClusters = np.arange(0,dataLength)[clusterAssignments == clusterAssignments[i]]
				orderedClusterAssignments[sameClusters] = currentClusterAssignment
				clusterCounter[sameClusters, currentClusterAssignment] +=1
				currentClusterAssignment = currentClusterAssignment - 1
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data / Find Cluster Statistics / Fill Transition Matrix
			previousCluster = 0
			for i in range(1,dataLength):
				currentCluster = orderedClusterAssignments[i]
				transitionCounter[previousCluster, currentCluster] += 1
				previousCluster = currentCluster

		transitionCounter = np.divide(transitionCounter, numIterations*(dataLength-1)*1.0)
		clusterCounter = np.divide(clusterCounter, numIterations*1.0)
		return {'transitionMatrix' : transitionCounter,
				'clusterAssignments' : clusterCounter}

	def stripDataSet(self,	inDataSet, dateColumn, indexColumn):
		if dateColumn is None:
			dateColumn = inDataSet.shape[1] #Basically make it out of range so it's not found
		return np.delete(inDataSet,[dateColumn,indexColumn],1)