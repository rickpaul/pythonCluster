#TODO: use sklearn.cluster to use other cluster types
import scipy.cluster.vq as spc
import numpy as np


class ClusterMeanPredictionAlgorithm:

	periodsAhead = np.array([1, 2, 3, 4, 5, 6, 9, 12, 18, 24, 36, 60, 120])

	def __init__(self,dataSet, indexColumn, dateColumn = None):
		self.dataSet = dataSet
		self.indexColumn = indexColumn
		self.dateColumn = dateColumn

	# Complex Distributions Version (produces weighted distribution for histogram output)
	# Dissatisfied with the weighting scheme...
	def performCMPAlgorithm_Weighted(self, specificDataPointIndex, numIterations = 200, numClusters = 4, subDataRatio = 0.5):
		#numClusters is based, for now, on 4 stages of the business cycle. We can make this stochastic.
		#We can also change this based on a verification algorithm (i.e. use test data to test training data)
		strippedDataSet = self.stripDataSet(self.dataSet, self.dateColumn, self.indexColumn)
		dataLength = strippedDataSet.shape[0]
		dataWidth = strippedDataSet.shape[1]
		specificDataPoint = strippedDataSet[specificDataPointIndex,:]

		numPeriods = len(self.periodsAhead)
		statistics = np.empty(shape=(numIterations, numPeriods, dataWidth),dtype=float)

		maxClusteringCost = -float("inf")
		minClusteringCost = float("inf")
		statisticWeightsbyIteration = np.empty(shape=(numIterations, 1),dtype=float)
		statisticWeightsbyPeriod = np.zeros(shape=(numIterations, numPeriods),dtype=float)
		statisticCountsbyPeriod = np.zeros(shape=(numPeriods, 1),dtype=float)

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
			maxClusteringCost = clusteringCost if clusteringCost > maxClusteringCost else maxClusteringCost
			minClusteringCost = clusteringCost if clusteringCost < minClusteringCost else minClusteringCost
			statisticWeightsbyIteration[i] = clusteringCost
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data
			allClusters = spc.vq(strippedDataSet, clusterCenters)
			clusterAssignments = allClusters[0]
			clusterDistortions = allClusters[1]
			# Perform Bootstrapped Clustering / Find Cluster of Specific Data Point
			specificCluster = clusterAssignments[specificDataPointIndex]
			dataPointWeight1 = clusterDistortions[specificDataPointIndex] #Weight 1: How representative is cluster of specific data point?
			# Perform Bootstrapped Clustering / Find Co-Clustered Data Points to Specific Data Point
			specificDataPointRelatives = subDataSetIndexes[clusterAssignments[subDataSetIndexes] == specificCluster]
			numSpecificDataPointRelatives = len(specificDataPointRelatives)
			# Perform Bootstrapped Clustering / Iterate over Related Data Points
			for relatedDataPointIndex in specificDataPointRelatives:
				# Perform Bootstrapped Clustering / Iterate over Related Data Points / Find Weight Statistics of Relative
				dataPointWeight2 = clusterDistortions[relatedDataPointIndex] #Weight 2: How representative is cluster of related data point?
				# Perform Bootstrapped Clustering / Iterate over Related Data Points / Find Statistics of Period Ahead
				for periodIndex, periodAhead in enumerate(self.periodsAhead):
					foundPeriod = periodAhead + relatedDataPointIndex
					if foundPeriod > (dataLength - 1):
						continue #no data to be gleaned
					futureCluster = clusterAssignments[foundPeriod]
					statistics[i,periodIndex,:] = clusterCenters[futureCluster,:]
					dataPointWeight3 = clusterDistortions[foundPeriod] #Weight 3: How representative is future's cluster of future data point?
					statisticWeightsbyPeriod[i,periodIndex] += ((1/(1+dataPointWeight1))*	# Between 0 and 1, how close is specific point to ORIGIN cluster center? Higher is better.
																(1/(1+dataPointWeight2)/numSpecificDataPointRelatives)* # Between 0 and 1, how close is AVERAGE related point to ORIGIN cluster center? Higher is better.
																(1/(1+dataPointWeight3))) # Between 0 and 1, how close is future point to its cluster center? Higher is better.
					statisticCountsbyPeriod[periodIndex] += 1
		validPeriods = np.where(statisticCountsbyPeriod >= 1)[0]
		# print statisticWeightsbyIteration #TEST
		# print statisticWeightsbyPeriod #TEST
		# print maxClusteringCost #TEST
		# print minClusteringCost #TEST
		return {'statistics' : statistics[:,validPeriods,:],
				'validPeriods' : validPeriods,
				'statisticWeightsbyPeriod' : statisticWeightsbyPeriod[:,validPeriods],
				'statisticWeightsbyIteration' : statisticWeightsbyIteration}

	# Simple Distributions Version (produces unweighted distribution for histogram output)
	def performCMPAlgorithm_Unweighted(self, specificDataPointIndex, numIterations = 200, numClusters = 4, subDataRatio = 0.5):
		#numClusters is based, for now, on 4 stages of the business cycle. We can make this stochastic.
		#We can also change this based on a verification algorithm (i.e. use test data to test training data)
		strippedDataSet = self.stripDataSet(self.dataSet, self.dateColumn, self.indexColumn)
		dataLength = strippedDataSet.shape[0]
		dataWidth = strippedDataSet.shape[1]
		specificDataPoint = strippedDataSet[specificDataPointIndex,:]

		numPeriods = len(self.periodsAhead)
		statistics = np.empty(shape=(numIterations, numPeriods, dataWidth),dtype=float)
		statisticWeights = np.zeros(shape=(numPeriods),dtype=float)

		# Perform Bootstrapped Clustering
		for i in range(0,numIterations):
			# Perform Bootstrapped Clustering / Chooose Data Subset
			subDataSetIndexes = np.random.choice(range(0,dataLength),size=dataLength*subDataRatio,replace=True) 
			subDataSet = strippedDataSet[subDataSetIndexes,:]
			# Perform Bootstrapped Clustering / Find Data Clusters for Subset of Data
			kMeansClusters = spc.kmeans(subDataSet, numClusters)
			clusterCenters = kMeansClusters[0]
			clusteringCost = kMeansClusters[1]
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data
			allClusters = spc.vq(strippedDataSet, clusterCenters)
			clusterAssignments = allClusters[0]
			clusterDistortions = allClusters[1]
			# Perform Bootstrapped Clustering / Find Cluster of Specific Data Point
			specificCluster = clusterAssignments[specificDataPointIndex]
			# Perform Bootstrapped Clustering / Find Co-Clustered Data Points to Specific Data Point
			specificDataPointRelatives = subDataSetIndexes[clusterAssignments[subDataSetIndexes] == specificCluster]
			# Perform Bootstrapped Clustering / Iterate over Related Data Points
			for relatedDataPointIndex in specificDataPointRelatives:
				# Perform Bootstrapped Clustering / Iterate over Related Data Points / Find Statistics of Period Ahead
				for periodIndex, periodAhead in enumerate(self.periodsAhead):
					foundPeriod = periodAhead + relatedDataPointIndex
					if foundPeriod > (dataLength - 1):
						continue #no data to be gleaned
					futureCluster = clusterAssignments[foundPeriod]
					statistics[i,periodIndex,:] = clusterCenters[futureCluster,:]
					statisticWeights[periodIndex] += 1
		validPeriods = np.where(statisticWeights >= 1)[0]
		return {'statistics' : statistics[:,validPeriods,:],
				'validPeriods' : validPeriods}
				# 'statisticWeightsbyPeriod' : np.ones(shape=(numIterations, len(validPeriods)),dtype=float),
				# 'statisticWeightsbyIteration' : np.ones(shape=(numIterations, numPeriods),dtype=float)}

	# Super Simple Flat Results Version (i.e. it produces simple averages of final statistics)
	def performCMPAlgorithm_Naive(self, specificDataPointIndex, numIterations = 200, nu,,,,,,,,,,,lusters = 4, subDataRatio = 0.5):
		#numClusters is based, for now, on 4 stages of the business cycle. We can make this stochastic.
		#We can also change this based on a verification algorithm (i.e. use test data to test training data)
		strippedDataSet = self.stripDataSet(self.dataSet, self.dateColumn, self.indexColumn)
		dataLength = strippedDataSet.shape[0]
		dataWidth = strippedDataSet.shape[1]
		specificDataPoint = strippedDataSet[specificDataPointIndex,:]


		numPeriods = len(self.periodsAhead)
		statistics = np.zeros(shape=(numPeriods, dataWidth),dtype=float)
		statisticWeights = np.zeros(shape=(numPeriods),dtype=float)

		# Perform Bootstrapped Clustering
		for i in range(0,numIterations):
			# Perform Bootstrapped Clustering / Chooose Data Subset
			subDataSetIndexes = np.random.choice(range(0,dataLength),size=dataLength*subDataRatio,replace=True) 
			subDataSet = strippedDataSet[subDataSetIndexes]
			# Perform Bootstrapped Clustering / Find Data Clusters for Subset of Data
			kMeansClusters = spc.kmeans(subDataSet, numClusters)
			clusterCenters = kMeansClusters[0]
			clusteringCost = kMeansClusters[1]
			# Perform Bootstrapped Clustering / Apply Found Data Clusters to All Data
			allClusters = spc.vq(strippedDataSet, clusterCenters)
			clusterAssignments = allClusters[0]
			clusterDistortions = allClusters[1]
			# Perform Bootstrapped Clustering / Find Cluster of Specific Data Point
			specificCluster = clusterAssignments[specificDataPointIndex]
			# Perform Bootstrapped Clustering / Find Co-Clustered Data Points to Specific Data Point
			specificDataPointRelatives = subDataSetIndexes[clusterAssignments[subDataSetIndexes] == specificCluster]
			# Perform Bootstrapped Clustering / Iterate over Related Data Points
			for relatedDataPointIndex in specificDataPointRelatives:
				# Perform Bootstrapped Clustering / Iterate over Related Data Points / Find Statistics of Period Ahead
				for periodIndex, periodAhead in enumerate(self.periodsAhead):
					foundPeriod = periodAhead + relatedDataPointIndex
					if foundPeriod > (dataLength - 1):
						continue #no data to be gleaned
					futureCluster = clusterAssignments[foundPeriod]
					statistics[periodIndex,:] += clusterCenters[futureCluster,:]
					statisticWeights[periodIndex] += 1 # TODO: We can weight this by the strength of the clustering and the distance of the relevant points to their relavant cluster centers
		return np.divide(statistics, statisticWeights)

	def stripDataSet(self,	inDataSet, dateColumn, indexColumn):
		if dateColumn is None:
			dateColumn = inDataSet.shape[1] #Basically make it out of range so it's not found
		return np.delete(inDataSet,[dateColumn,indexColumn],1)