from timeit import default_timer as timer
import numpy as np
from copy import deepcopy
from math import log
from collections import Counter
from operator import itemgetter
import pudb # TEST
import pprint # TEST
# import scipy.optimize.minimize as minimize
# import scipy.optimize.OptimizeResult as minResult

# See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult

class DataCompatabilityError(Exception):
	# TODO: Why isn't this displaying correctly?
	def __init__(self, variableName, variableLength, expectedVariableLength):
		self.variableName = variableName
		self.variableLength = variableLength
		self.expectedVariableLength = expectedVariableLength
	def __str__(self):
		print variableName + ' expected dimension ' + str(expectedVariableLength) + ', found ' + str(variableLength)

class NoDataError(Exception):
	pass

class BrownClusterModel:
	TaskStack = []

	START_CLUSTER_SYMBOL = -1
	NO_CLUSTER_SYMBOL = -2

	def __init__(self):
		self.wordSequence = []
		self.sequenceLength = 0

    #################### Data Addition Methods
    #TODO: Make more efficient. (You don't really need to redo the statistics every time you add data)
	def addTrainingData(self, wordSequence, verbose=False):
		(wordDataLength, wordDataWidth) = wordSequence.shape #Assume it comes in as numpy array
		# Data Validation / Word Data Width
		if not hasattr(self, 'wordDataWidth'):
			self.wordDataWidth = wordDataWidth
		if wordDataWidth != self.wordDataWidth:
			raise DataCompatabilityError('wordDataWidth', wordDataWidth, self.wordDataWidth)
		# Add Data. Simple Concatenation
		self.wordSequence = self.wordSequence + self.tuplefyWordSequence(wordSequence)
		# Update Sequence Length
		self.sequenceLength += wordDataLength
		# Reset Variables and Flags
		self.resetDataStatistics(verbose=verbose)
		if verbose:
			print 'Added new word sequence of length ' + str(wordDataLength)
			print 'Total word sequence is now ' + str(self.sequenceLength)

	def tuplefyWordSequence(self, wordSequence):
		wordDataLength = len(wordSequence)
		newWordSequence = [None] * wordDataLength #Pre-allocate. Probably not worthwhile...
		for i in range(wordDataLength):
			newWordSequence[i] = tuple(wordSequence[i,:])
		return newWordSequence

	def PerformSingleTask(self, f):
		# Todo: Print out what it's doing.
		return f() #in case f() returns something

	# Returns: 	True if there are tasks left to perform. 
	#			False if there are no tasks left to perform.
	def performTaskStack_NextStep(self, verbose=False):
		if len(self.TaskStack) == 0:
			if verbose:
				print 'No Tasks Remain in TaskStack.'
			return False
		else:
			fn = self.TaskStack.pop()
			if verbose:
				print 'Performing next task from TaskStack: '
			self.PerformSingleTask(fn)
			return True

	def clearTaskStack(self, verbose=False):
		if verbose:
			print 'Clearing TaskStack...'
		while self.performTaskStack_NextStep():
			pass

	def resetDataStatistics(self, verbose=False):
		# (Superfluous) Delete Existing Data
		self.sortedWords = []
		self.wordClusterMapping = {}
		self.clusterSequence = []
		self.clusters = []
		self.clusterNGramCounts = {}
		self.clusterCostTable = {}
		self.clusterMergeCostReductionTable = {} 

		# Add New Tasks
		self.TaskStack = []
		self.TaskStack.append(lambda: self.defineClusterMergeCostReductionTable(verbose=verbose))
		self.TaskStack.append(lambda: self.defineClusterCostTable(verbose=verbose))
		self.TaskStack.append(lambda: self.defineClusterCounts(verbose=verbose))
		self.TaskStack.append(lambda: self.defineClusterSequence(verbose=verbose))
		self.TaskStack.append(lambda: self.defineWordClusterMap(verbose=verbose))

		# Clear The Tasks
		self.clearTaskStack(verbose=verbose)

	def defineWordClusterMap(self, numInitialClusters=None, verbose=False):
		if verbose:
			print '...Defining Word-to-Cluster Mapping...'
		start = timer()
		self.sortedWords = [x[0] for x in Counter(self.wordSequence).most_common()]
		if numInitialClusters is None:
			self.wordClusterMapping = dict(zip(self.sortedWords,range(0,len(self.sortedWords))))
		else:
			self.wordClusterMapping = dict(zip(self.sortedWords[0:numInitialClusters],range(0,numInitialClusters)))
		end = timer()
		if verbose:
			print "Defined Word Cluster Map for  " + str(self.sequenceLength) + " words."
			print "\t" + str(len(self.sortedWords)) + " words found."
			print "\t" + str(len(self.wordClusterMapping)) + " clusters created."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	def defineClusterCounts(self, verbose=False):
		if verbose:	
			print '...Counting Cluster Unigrams and Bigrams from Cluster Sequence...'	
		start = timer()
		self.clusterNGramCounts = self.defineClusterCounts_generic(self.clusterSequence, verbose=verbose)
		self.clusters = self.clusterNGramCounts[0].keys()
		end = timer()
		if verbose:
			print "Found cluster sequence for " + str(sequenceLength-1) + " words."
			print "\t" + str(len(clusterNGramCounts[0])) + " clusters were found."
			print "\t" + str(len(clusterNGramCounts[1])) + " bigrams were found."
			if self.NO_CLUSTER_SYMBOL in clusterNGramCounts[0]:
				print "\tUnclustered words remain in the dataset."
			else:
				print "\tNo unclustered words remain in the dataset."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	def defineClusterCounts_generic(self, clusterSequence, verbose=False):
		# clusterSequence = [self.START_CLUSTER_SYMBOL] + clusterSequence
		sequenceLength = len(clusterSequence)
		clusterNGramCounts = {}
		clusterNGramCounts[0] = {}
		clusterNGramCounts[1] = {}
		clusterNGramCounts[0][clusterSequence[0]] = 1.0
		for i in range(1,sequenceLength):
			cluster = clusterSequence[i]
			bigram = (clusterSequence[i-1], cluster)
			clusterNGramCounts[0][cluster] = clusterNGramCounts[0].get(cluster, 0.0) + 1.0
			clusterNGramCounts[1][bigram] = clusterNGramCounts[1].get(bigram, 0.0) + 1.0
			#CONSIDER: Does the concept of discounted probabilities mean anything here?
		return clusterNGramCounts

	def defineClusterSequence(self, verbose=False):
		if verbose:	
			print '...Creating Cluster Sequence from Word-to-Cluster Mapping and Word Sequence...'
		start = timer()
		self.clusterSequence = self.defineClusterSequence_generic(self.wordSequence, verbose=verbose)
		end = timer()
		if verbose:
			print "Found cluster sequence for " + str(sequenceLength) + " words."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	def defineClusterSequence_generic(self, wordSequence, verbose=False):
		sequenceLength = len(wordSequence)
		clusterSequence = [None] * sequenceLength
		for i in range(sequenceLength):
			clusterSequence[i] = self.wordClusterMapping.get(wordSequence[i], self.NO_CLUSTER_SYMBOL)
		return clusterSequence

	def defineClusterCostTable(self, verbose=False):
		if verbose:	
			print '...Creating Cluster Cost Table (Finding Bigram Graph Edge Costs)...'		
		start = timer()
		numClusters = len(self.clusters)
		for i in range(numClusters):
			cluster1 = self.clusters[i]
			cluster1Count = self.getClusterCount(cluster1)
			for j in range(i+1):
				if i == j:
					bigramCount = self.getBigramCount(cluster1, cluster1)
					cost = self.findMutualInformation(bigramCount, cluster1Count, cluster1Count)
					self.clusterCostTable[(cluster1,cluster1)] = cost
				else:
					cluster2 = self.clusters[j]
					cluster2Count = self.getClusterCount(cluster2)
					bigram1Count = self.getBigramCount(cluster1, cluster2)
					bigram2Count = self.getBigramCount(cluster2, cluster1)
					cost = self.findMutualInformation(bigram1Count, cluster1Count, cluster2Count)
					cost += self.findMutualInformation(bigram2Count, cluster1Count, cluster2Count)
					self.clusterCostTable[(cluster1,cluster2)] = cost
		end = timer()
		if verbose:
			print "Defined Cluster Cost Table."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	def getClusterCostBigram(self, cluster1, cluster2):
		# Assumes one of the two is in the table...
		return (cluster1, cluster2) if (cluster1, cluster2) in self.clusterCostTable else (cluster2, cluster1)

	def getClusterCost(self, cluster1, cluster2):
		return self.clusterCostTable[self.getClusterCostBigram(cluster1, cluster2)]

	def findMutualInformation(self, bigramCount, unigram1Count, unigram2Count, sequenceLength=None):
		if sequenceLength is None:
			sequenceLength = self.sequenceLength
		if bigramCount > 0:
			# I've been uncomfortable with logs being above/below 0, but what I'm coming to realize is that the zero
			# intercept of mutual information is where information goes from useful to not..?
			# Sample probability? or Population probability? Population.
			# n = sequenceLength - 1 # Sample probability
			n = sequenceLength # Population probability
			return bigramCount/n * log(n*bigramCount/unigram1Count/unigram2Count, 2)
		else:
			if unigram1Count == 0 or unigram2Count == 0:
				raise Exception('Erroneous clusters')
			return 0.0

	# This is the mutual information from treating two separate clusters as one.
	# If you want to treat 1 and 2 as A, vis-a-vis 3, then:
	# you need to account for 1->3 connections, 2->3, 3->1, and 3->2, and treat them all as A->3 and 3->A;
	# you also need to account for P(A) = P(1)+P(2)
	def findMutualInformationOfMergedClusters_OtherConnection(	self, 
																mergeCluster1, 
																mergeCluster2,
																otherCluster,
																mergeCluster1Count=None, 
																mergeCluster2Count=None, 
																otherClusterCount=None,
																sequenceLength=None):
		if mergeCluster1Count is None:
			mergeCluster1Count = self.getClusterCount(mergeCluster1)
		if mergeCluster2Count is None:
			mergeCluster2Count = self.getClusterCount(mergeCluster2)
		if otherClusterCount is None:
			otherClusterCount = self.getClusterCount(otherCluster)
		# Consider: we could just combine the two bigram1 and two bigram2 counts
		# 			(it won't change the mutual information equation)
		mutualInformation = 0.0
		bigram1Count = self.getBigramCount(otherCluster, mergeCluster1)
		bigram2Count = self.getBigramCount(otherCluster, mergeCluster2)
		mutualInformation += self.findMutualInformation(	(bigram1Count + bigram2Count),
															otherClusterCount,
															(mergeCluster1Count+mergeCluster2Count),
															sequenceLength
														)
		bigram1Count = self.getBigramCount(mergeCluster1, otherCluster)
		bigram2Count = self.getBigramCount(mergeCluster2, otherCluster)
		mutualInformation += self.findMutualInformation(	(bigram1Count + bigram2Count),
															otherClusterCount,
															(mergeCluster1Count+mergeCluster2Count),
															sequenceLength
														)
		return mutualInformation

	# This is the mutual information from treating two separate clusters as one.
	# If you want to treat 1 and 2 as A, then:
	# you need to account for 1->2 connections, 2->1, 1->1, and 2->2, and treat them all as A->A, and
	# you need to account for P(A) = P(1)+P(2)
	def findMutualInformationOfMergedClusters_SelfConnection(	self, 
																mergeCluster1, 
																mergeCluster2, 
																mergeCluster1Count=None, 
																mergeCluster2Count=None, 
																sequenceLength=None):
		if mergeCluster1Count is None:
			mergeCluster1Count = self.getClusterCount(mergeCluster1)
		if mergeCluster2Count is None:
			mergeCluster2Count = self.getClusterCount(mergeCluster2)
		bigram1Count = self.getBigramCount(mergeCluster1, mergeCluster2)
		bigram2Count = self.getBigramCount(mergeCluster2, mergeCluster1)
		bigram3Count = self.getBigramCount(mergeCluster1, mergeCluster1)
		bigram4Count = self.getBigramCount(mergeCluster2, mergeCluster2)
		totalBigramCount = bigram1Count + bigram2Count + bigram3Count + bigram4Count
		return self.findMutualInformation(	totalBigramCount,
											(mergeCluster1Count+mergeCluster2Count),
											(mergeCluster1Count+mergeCluster2Count), 
											sequenceLength
										)

	def findMutualInformationOfMergedClusters_PairPairConnection(	self, 
																	cluster1,
																	cluster2,
																	cluster3, 
																	cluster4,
																	cluster1Count=None, 
																	cluster2Count=None, 
																	cluster3Count=None, 
																	cluster4Count=None, 
																	sequenceLength=None):
		raise NotImplementedError('Not Tested Yet')
		if cluster1Count is None:
			cluster1Count = self.getClusterCount(cluster1)
		if cluster2Count is None:
			cluster2Count = self.getClusterCount(cluster2)
		if cluster3Count is None:
			cluster3Count = self.getClusterCount(cluster3)
		if cluster4Count is None:
			cluster4Count = self.getClusterCount(cluster4)
		mutualInformation = 0.0
		 # Group 1 to Group 2
		bigram1Count = self.getBigramCount(cluster1, cluster3)
		bigram2Count = self.getBigramCount(cluster1, cluster4)
		bigram3Count = self.getBigramCount(cluster2, cluster3)
		bigram4Count = self.getBigramCount(cluster2, cluster4)
		totalBigramCount = bigram1Count + bigram2Count + bigram3Count + bigram4Count
		mutualInformation += self.findMutualInformation(	totalBigramCount,
															(cluster1Count+cluster2Count),
															(cluster3Count+cluster4Count), 
															sequenceLength
														)
		# Group 2 to Group 1
		bigram1Count = self.getBigramCount(cluster3, cluster1)
		bigram2Count = self.getBigramCount(cluster3, cluster2)
		bigram3Count = self.getBigramCount(cluster4, cluster1)
		bigram4Count = self.getBigramCount(cluster4, cluster2)
		totalBigramCount = bigram1Count + bigram2Count + bigram3Count + bigram4Count
		mutualInformation += self.findMutualInformation(	totalBigramCount,
															(cluster1Count+cluster2Count),
															(cluster3Count+cluster4Count), 
															sequenceLength
														)
		return mutualInformation

	# This function gives the merge reduction cost for a single cluster pair using the naive algorithm.
	# This algorithm has been verified. 
	def defineClusterMergeCostReductionTable_singleClusterPair(self, cluster1, cluster2, verbose=False):
		sequenceLength = self.sequenceLength
		cluster1Count = self.getClusterCount(cluster1)
		cluster2Count = self.getClusterCount(cluster2)
		clusterCostReduction = 0.0
		clusterCostAddition = 0.0
		for cluster3 in self.clusters:
			if cluster3 == cluster1 or cluster3 == cluster2:
				continue #deal with these separately. 
			clusterCostReduction += self.getClusterCost(cluster1, cluster3)  # Encompasses 1->3 and 3->1
			clusterCostReduction += self.getClusterCost(cluster2, cluster3)  # Encompasses 2->3 and 3->2
			# This is the procedure you get if you try to combine two nodes into one:
			# P(c,c')*log(P(c,c')/P(c)/P(c'))
			cluster3Count = self.getClusterCount(cluster3)
			clusterCostAddition += self.findMutualInformationOfMergedClusters_OtherConnection(	cluster1, cluster2, cluster3,
																								cluster1Count, cluster2Count, cluster3Count,
																								sequenceLength)
		clusterCostReduction += self.getClusterCost(cluster1, cluster2) # Encompasses 1->2 and 2->1
		clusterCostReduction += self.getClusterCost(cluster1, cluster1) # Encompasses 1->1
		clusterCostReduction += self.getClusterCost(cluster2, cluster2) # Encompasses 2->2
		clusterCostAddition += self.findMutualInformationOfMergedClusters_SelfConnection(	cluster1, cluster2, 
																							cluster1Count, cluster2Count, 
																							sequenceLength)
		return (clusterCostAddition - clusterCostReduction)

	def defineClusterMergeCostReductionTable(self, verbose=False):
		start = timer()
		# Consider: Should we store this separately? (i.e. clusters = self.clusters)
		clusters = self.clusters
		numClusters = len(self.clusters)
		for i in range(numClusters):
			cluster1 = self.clusters[i]
			for j in range(i):
				cluster2 = self.clusters[j]
				mergeCost = self.defineClusterMergeCostReductionTable_singleClusterPair(cluster1, cluster2, verbose=verbose)
				self.clusterMergeCostReductionTable[self.getClusterCostBigram(cluster1, cluster2)] = mergeCost
		end = timer()
		if verbose:
			print "Defined Cluster Merge Cost Reduction Table."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"


	def updateMergeCostReductionTable_UnmergedClusters(self, cluster1, cluster2, mergeCluster1, mergeCluster2, verbose=False):
		cluster1Count = self.getClusterCount(cluster1)
		cluster2Count = self.getClusterCount(cluster2)
		mergeCluster1Count = self.getClusterCount(mergeCluster1)
		mergeCluster2Count = self.getClusterCount(mergeCluster2)
		currentMergeCost = self.clusterMergeCostReductionTable[self.getClusterCostBigram(cluster1, cluster2)] 
		mergeCostAddition = 0
		mergeCostAddition += self.getClusterCost(cluster1, mergeCluster1)
		mergeCostAddition += self.getClusterCost(cluster1, mergeCluster2)
		mergeCostAddition += self.getClusterCost(cluster2, mergeCluster1)
		mergeCostAddition += self.getClusterCost(cluster2, mergeCluster2)
		mergeCostAddition += self.findMutualInformationOfMergedClusters_PairPairConnection(	cluster1, 
																							cluster2, 
																							mergeCluster1, 
																							mergeCluster2,
																							cluster1Count,
																							cluster2Count,
																							mergeCluster1Count,
																							mergeCluster2Count)
		mergeCostReduction = 0
		mergeCostReduction += self.findMutualInformationOfMergedClusters_OtherConnection(mergeCluster1, mergeCluster2, cluster1, mergeCluster1Count, mergeCluster2Count, cluster1Count) # 1 -> (m1+m2)
		mergeCostReduction += self.findMutualInformationOfMergedClusters_OtherConnection(mergeCluster1, mergeCluster2, cluster2, mergeCluster1Count, mergeCluster2Count, cluster2Count) # 2 -> (m1+m2)
		mergeCostReduction += self.findMutualInformationOfMergedClusters_OtherConnection(cluster1, cluster2, mergeCluster1, cluster1Count, cluster2Count, mergeCluster1Count) # m1 -> (1+2)
		mergeCostReduction += self.findMutualInformationOfMergedClusters_OtherConnection(cluster1, cluster2, mergeCluster2, cluster1Count, cluster2Count, mergeCluster2Count) # m2 -> (1+2)
		return (mergeCostAddition - mergeCostReduction)

	def updateMergeCostReductionTable(self, mergeCluster1, mergeCluster2, verbose=False):
		raise NotImplementedError
		#TODO: We need to make sure we've updated clusterNGramCounts first
		clusters = self.clusters
		numClusters = len(self.clusters)
		for i in range(numClusters):
			cluster1 = self.clusters[i]
			for j in range(i):
				cluster2 = self.clusters[j]
				if (cluster1 == mergeCluster1 or cluster1 == mergeCluster2 or cluster2 == mergeCluster1 or cluster2 == mergeCluster2):
					# TODO: Right?
					self.defineClusterMergeCostReductionTable_singleClusterPair(cluster1, cluster2, verbose=verbose)
				else:
					pass

	def findMergeClusters(self):
		if len(self.clusterMergeCostReductionTable) == 0: # Necessary?
			return (False, None) # Necessary?
		return (True, sorted(self.clusterMergeCostReductionTable.items(), key=itemgetter(1))[0][0])

	def getBigramCount(self, cluster1, cluster2):
		return self.clusterNGramCounts[1].get((cluster1, cluster2), 0.0)

	def getClusterCount(self, cluster1):
		return self.clusterNGramCounts[0][cluster1] # Throw an error if not found

	def removeBigramCount(self, cluster1, cluster2):
		if (cluster1, cluster2) in self.clusterNGramCounts[1]:
			count = self.clusterNGramCounts[1][(cluster1, cluster2)]
			del self.clusterNGramCounts[1][(cluster1, cluster2)]
			return count
		else:
			return 0

	def contributeBigramCount(self, cluster1, cluster2, addCount):
		if (cluster1, cluster1) in self.clusterNGramCounts[1]:
			self.clusterNGramCounts[1][(cluster1, cluster2)] += addCount
		else:
			self.clusterNGramCounts[1][(cluster1, cluster2)] = addCount

	def mergeClusters(self):
		raise NotImplementedError
		# 1) Find Merge Clusters
		(success, mergeClusters) = self.findMergeClusters()
		if not success:
			return False
		(cluster1, cluster2) = mergeClusters
		# 2) Change Cluster Counts
		cluster1Count = self.getClusterCount(cluster1)
		cluster2Count = self.getClusterCount(cluster2)
		self.clusterNGramCounts[0][cluster1] += cluster2Count
		# Change Cluster Counts / Remove Old Cluster
		del self.clusterNGramCounts[0][cluster2]
		# Change Cluster Counts / Reset Clusters
		self.clusters = self.clusterNGramCounts[0].keys()
		# 3) Change Bigram Counts and ClusterCost Table
		# Change Bigram Counts and ClusterCost Table / Change Non-Merging Clusters
		for cluster3 in self.clusters:
			if cluster3 == cluster1 or cluster3 == cluster2:
				continue #deal with these separately. 
			# Change Bigram Counts and ClusterCost Table / Change Non-Merging Clusters / Change Bigram Counts
			self.contributeBigramCount(cluster3, cluster1, self.removeBigramCount(cluster3, cluster2)) # 3->2 => 3->1
			self.contributeBigramCount(cluster1, cluster3, self.removeBigramCount(cluster2, cluster3)) # 2->3 => 1->3
			# Change Bigram Counts and ClusterCost Table / Change Non-Merging Clusters / Change ClusterCost Table
			cluster3Count = self.getClusterCount(cluster3)
			raise NotImplementedError
		
		self.clusterNGramCounts[1][(cluster1,cluster1)] += cluster2Count
		# Change Bigram Counts and ClusterCost Table / Change Merging Clusters / Change Bigram Counts
		# Don't need to do 1->1
		bigramCount = 0
		bigramCount += self.removeBigramCount(cluster2, cluster1) # 2->1 => 1->1
		bigramCount += self.removeBigramCount(cluster1, cluster2) # 1->2 => 1->1
		bigramCount += self.removeBigramCount(cluster2, cluster2) # 2->2 => 1->1
		# all => 1->1
		self.contributeBigramCount(cluster1, cluster1, bigramCount)
		# Change Bigram Counts and ClusterCost Table / Change Merging Clusters / Change ClusterCost Table
		raise NotImplementedError




		return True


	# # TODO: Why are we passing in clusterNGram Counts? Why not self.clusterNGramCounts?
	# #		Is it because we want to keep it generic for trying out new versions?
	# def findClusteringCost(self, clusterNGramCounts, sequenceLength=None, verbose=False):
	# 	start = timer()
	# 	if sequenceLength is None:
	# 		sequenceLength = self.sequenceLength
	# 	# Consider: Should we store this separately? (i.e. clusters = self.clusters)
	# 	clusters = clusterNGramCounts[0].keys()
	# 	qualityCost = 0
	# 	for cluster1 in clusters:
	# 		cluster1Count = clusterNGramCounts[0].get(cluster1, 0)
	# 		for cluster2 in clusters:
	# 			cluster2Count = clusterNGramCounts[0].get(cluster2, 0)
	# 			bigramCount = clusterNGramCounts[1].get((cluster1, cluster2),0)
	# 			qualityCost += self.findMutualInformation(bigramCount, cluster1Count, cluster2Count)
	# 	end = timer()
	# 	if verbose:
	# 		print "Found Clustering Cost."
	# 		print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"
	# 	return qualityCost

	# def findBrownClustering(self, numInitialClusters=100):
	# 	raise NotImplementedError
	# 	self.defineWordClusterMap(self, numInitialClusters=numInitialClusters)
	# 	clusterSequence = self.defineClusterSequence()
	# 	clusterNGramCounts = self.defineClusterCounts(clusterSequence)
	# 	nextWordPointer = numInitialClusters
	# 	unclusteredWords = self.NO_CLUSTER_SYMBOL in clusterNGramCounts[0]
	# 	unmergedClusters = len(clusterNGramCounts) > 1
	# 	while unclusteredWords or unmergedClusters:
	# 		if unclusteredWords:
	# 			changeWord = self.sortedWords[nextWordPointer]
	# 			(clusterSequence, clusterNGramCounts) = self.changeWordCluster(	clusterSequence,
	# 																			clusterNGramCounts,
	# 																			changeWord,
	# 																			nextWordPointer,
	# 																			oldCluster=self.NO_CLUSTER_SYMBOL)
	# 			nextWordPointer += 1
	# 			unclusteredWords = self.NO_CLUSTER_SYMBOL in clusterNGramCounts[0]
	# 		if unmergedClusters:
	# 			raise NotImplementedError
	# 			unmergedClusters = len(clusterNGramCounts) > 1

	# def changeCluster(self, clusterNGramCounts, clusterSequence, newCluster, oldCluster):
	# 	raise NotImplementedError
	# 	# clusterSequence = [self.START_CLUSTER_SYMBOL] + clusterSequence # TODO: will have to make sure it's a copy!
	# 	sequenceLength = len(clusterSequence)
	# 	# For Each Cluster
	# 	for i in range(sequenceLength-1,0,-1):
	# 		# For Each Cluster / If Cluster is Changing Cluster
	# 		if clusterSequence[i] == oldCluster:
	# 			# For Each Cluster / If Cluster is Changing Cluster / Adjust Unigram Counts
	# 			clusterNGramCounts[0][newCluster] = clusterNGramCounts[0].get(newCluster,0) + 1
	# 			clusterNGramCounts[0][oldCluster] -= 1
	# 			if clusterNGramCounts[0][oldCluster] == 0:
	# 				del(clusterNGramCounts[0][oldCluster]) #delete it so it throws an error if we try to access it later.			
	# 			# For Each Cluster / If Cluster is Changing Cluster / Adjust Bigram Counts
	# 			newBigram = (clusterSequence[i-1],newCluster)
	# 			oldBigram = (clusterSequence[i-1],oldCluster)
	# 			clusterNGramCounts[1][newBigram] = clusterNGramCounts[1].get(newBigram,0) + 1
	# 			clusterNGramCounts[1][oldBigram] -= 1
	# 			if clusterNGramCounts[1][oldBigram] == 0:
	# 				del(clusterNGramCounts[1][oldBigram]) #delete it so it throws an error if we try to access it later.
	# 			# For Each Cluster / If Cluster is Changing Cluster / Adjust Cluster Sequence
	# 			clusterSequence[i] = newCluster	
	# 	return (clusterSequence, clusterNGramCounts)

	# def changeWordCluster(self, clusterNGramCounts, clusterSequence, changeWord, newCluster, oldCluster=None):
	# 	raise NotImplementedError
	# 	if oldCluster is None:
	# 		oldCluster = self.wordClusterMapping[changeWord]
	# 	# clusterSequence = [self.START_CLUSTER_SYMBOL] + clusterSequence
	# 	sequenceLength = len(clusterSequence)
	# 	# For Each Word
	# 	for i in range(sequenceLength-1,0,-1):
	# 		# For Each Word / If Word is Changing Word
	# 		if self.wordSequence[i-1] == changeWord:
	# 			# For Each Word / If Word is Changing Word / Adjust Unigram Counts
	# 			clusterNGramCounts[0][newCluster] = clusterNGramCounts[0].get(newCluster,0) + 1
	# 			clusterNGramCounts[0][oldCluster] -= 1
	# 			if clusterNGramCounts[0][oldCluster] == 0:
	# 				del(clusterNGramCounts[0][oldCluster]) #delete it so it throws an error if we try to access it later.			
	# 			# For Each Word / If Word is Changing Word / Adjust Bigram Counts
	# 			newBigram = (clusterSequence[i-1],newCluster)
	# 			oldBigram = (clusterSequence[i-1],oldCluster)
	# 			clusterNGramCounts[1][newBigram] = clusterNGramCounts[1].get(newBigram,0) + 1
	# 			clusterNGramCounts[1][oldBigram] -= 1
	# 			if clusterNGramCounts[1][oldBigram] == 0:
	# 				del(clusterNGramCounts[1][oldBigram]) #delete it so it throws an error if we try to access it later.
	# 			# For Each Word / If Word is Changing Word / Adjust Cluster Sequence
	# 			clusterSequence[i] = newCluster
	# 	return (clusterSequence, clusterNGramCounts)
