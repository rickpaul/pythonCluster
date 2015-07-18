from timeit import default_timer as timer
import numpy as np
from copy import deepcopy
from math import log
from collections import Counter
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
	START_CLUSTER_SYMBOL = -1
	NO_CLUSTER_SYMBOL = -2

	def __init__(self):
		self.wordSequence = []
		self.clusterSequence = []
		self.sequenceLength = 0

    #################### Data Addition Methods
    #TODO: Make more efficient. You don't need to redo the statistics every time you add data
	def addTrainingData(self, wordSequence, verbose=False):
		(wordDataLength, wordDataWidth) = wordSequence.shape # Assume it comes in as numpy array
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
		self.wordClusterMapping = {}
		self.hasResetWordClusterMap = False
		self.hasResetClusterSequence = False
		self.clusterNGramCounts = {}
		self.hasResetClusterNGramCounts = False
		self.sortedWords = []
		self.hasResetSortedWords = False
		self.clusterCostTable = {}
		self.hasResetClusterCostTable = False
		self.clusterMergeCostReductionTable = {}
		self.hasResetClusterMergeCostReductionTable = False
		if verbose:
			print 'Added new word sequence of length ' + str(wordDataLength)
			print 'Total word sequence is now ' + str(self.sequenceLength)

	def tuplefyWordSequence(self, wordSequence):
		wordDataLength = len(wordSequence)
		newWordSequence = [None] * wordDataLength #Pre-allocate. Probably not worthwhile...
		for i in range(wordDataLength):
			newWordSequence[i] = tuple(wordSequence[i,:])
		return newWordSequence

    #################### Initial Data Statistics Methods
	def resetDataStatistics(self, finalStep=None, verbose=False):
    	# This method is probably more naturally done recursively. (Actually, it sort of is. Messily.) But this is more readable.
    	# TODO: Maybe reformat this.
    	# The data is reset in the following order.
    	# 1) WordSequence (implied. This requires a reset, actually)
		# 2) WordClusterMap (SortedWords, too) 
		# 3) ClusterSequence
		# 4) ClusterNGramCounts
		# 5) ClusterCostTable
		# 6) ClusterMergeCostReductionTable
		if self.sequenceLength == 0:
			raise NoDataError

		if verbose:
			print '...Defining Word-to-Cluster Mapping...'
		if (not self.hasResetSortedWords) or (not self.hasResetWordClusterMap):
			self.defineWordClusterMap(verbose=verbose)
		if finalStep == 'defineWordClusterMap':
			return

		if verbose:	
			print '...Transforming Word Sequence to Cluster Sequence...'
		if (not self.hasResetClusterSequence):
			self.defineClusterSequence(verbose=verbose)
		if finalStep == 'defineClusterSequence':
			return

		if verbose:	
			print '...Counting Cluster Unigrams and Bigrams from Cluster Sequence...'		
		if (not self.hasResetClusterNGramCounts):
			self.defineClusterCounts(verbose=verbose)
		if finalStep == 'defineClusterCounts':
			return

		if verbose:	
			print '...Creating Cluster Cost Table (Finding Bigram Graph Edge Costs)...'
		if (not self.hasResetClusterCostTable):
			self.defineClusterCostTable(verbose=verbose)
		if finalStep == 'defineClusterCostTable':
			return

		if verbose:
			print '...Creating Merge Cost Reduction Table (Using Bigram Graph Edge Costs)...'
		if (not self.hasResetClusterMergeCostReductionTable):
			self.defineClusterMergeCostReductionTable(verbose=verbose)
		if finalStep == 'defineClusterMergeCostReductionTable':
			return #Yes, this is superfluous. Symmetry.

	def defineWordClusterMap(self, numInitialClusters=None, verbose=False):
		start = timer()
		self.sortedWords = [x[0] for x in Counter(self.wordSequence).most_common()]
		if numInitialClusters is None:
			self.wordClusterMapping = dict(zip(self.sortedWords,range(0,len(self.sortedWords))))
		else:
			self.wordClusterMapping = dict(zip(self.sortedWords[0:numInitialClusters],range(0,numInitialClusters)))
		self.hasResetSortedWords = True
		self.hasResetWordClusterMap = True
		end = timer()
		if verbose:
			print "Defined Word Cluster Map for  " + str(self.sequenceLength) + " words."
			print "\t" + str(len(self.sortedWords)) + " words found."
			print "\t" + str(len(self.wordClusterMapping)) + " clusters created."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	def defineClusterSequence(self, verbose=False):
		#This check is done within the generic function as well. This is mostly just for symmetry.
		if not self.hasResetWordClusterMap: 
			self.resetDataStatistics('defineWordClusterMap', verbose=verbose)
		self.clusterSequence = self.defineClusterSequence_generic(self.wordSequence, verbose=verbose)
		self.hasResetClusterSequence = True
	
	def defineClusterCounts_generic(self, clusterSequence, verbose=False):
		start = timer()
		# clusterSequence = [self.START_CLUSTER_SYMBOL] + clusterSequence
		sequenceLength = len(clusterSequence)
		clusterNGramCounts = {}
		clusterNGramCounts[0] = {}
		clusterNGramCounts[1] = {}
		# for i in range(0,sequenceLength):
		clusterNGramCounts[0][clusterSequence[0]] = 1.0
		for i in range(1,sequenceLength):
			cluster = clusterSequence[i]
			bigram = (clusterSequence[i-1], cluster) #tuple
			clusterNGramCounts[0][cluster] = clusterNGramCounts[0].get(cluster,0.0) + 1.0
			clusterNGramCounts[1][bigram] = clusterNGramCounts[1].get(bigram,0.0) + 1.0
			#CONSIDER: Does the concept of discounted probabilities mean anything here?
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
		return clusterNGramCounts

	def defineClusterCounts(self, verbose=False):
		if not self.hasResetClusterSequence:
			self.resetDataStatistics('defineClusterSequence', verbose=verbose)
		self.clusterNGramCounts = self.defineClusterCounts_generic(self.clusterSequence, verbose=verbose)
		self.hasResetClusterNGramCounts = True

	def defineClusterSequence_generic(self, wordSequence, verbose=False):
		start = timer()
		if not self.hasResetWordClusterMap:
			self.resetDataStatistics('defineWordClusterMap', verbose=verbose)
		sequenceLength = len(wordSequence)
		clusterSequence = [None] * sequenceLength
		for i in range(sequenceLength):
			clusterSequence[i] = self.wordClusterMapping.get(wordSequence[i], self.NO_CLUSTER_SYMBOL)
		end = timer()
		if verbose:
			print "Found cluster sequence for " + str(sequenceLength) + " words."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"
		return clusterSequence

	def defineClusterCostTable(self, verbose=False):
		if not self.hasResetClusterNGramCounts:
			self.resetDataStatistics('defineClusterCounts', verbose=verbose)
		start = timer()
		clusters = self.clusterNGramCounts[0].keys()
		numClusters = len(clusters)
		for i in range(numClusters):
			cluster1 = clusters[i]
			cluster1Count = self.clusterNGramCounts[0].get(cluster1 , 0)
			for j in range(i+1):
				if i == j:
					bigramCount = self.clusterNGramCounts[1].get((cluster1, cluster1),0)
					# if bigramCount == 0:
					# 	cost = -float('inf')
					# else:
					cost = self.findMutualInformation(bigramCount, cluster1Count, cluster1Count)
					self.clusterCostTable[(cluster1,cluster1)] = cost
				else:
					cluster2 = clusters[j]
					cluster2Count = self.clusterNGramCounts[0].get(cluster2, 0)
					bigram1Count = self.clusterNGramCounts[1].get((cluster1, cluster2),0)
					bigram2Count = self.clusterNGramCounts[1].get((cluster2, cluster1),0)
					# if bigram1Count == 0 and bigram2Count == 0:
					# 	cost = -float('inf')
					# else:
					cost = self.findMutualInformation(bigram1Count, cluster1Count, cluster2Count)
					cost += self.findMutualInformation(bigram2Count, cluster1Count, cluster2Count)
					self.clusterCostTable[(cluster1,cluster2)] = cost
		end = timer()
		self.hasResetClusterCostTable = True
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
			return bigramCount/sequenceLength * log(bigramCount/unigram1Count/unigram2Count*sequenceLength, 2)
			# Sample probability? or Population probability?
			# return bigramCount/(sequenceLength-1) * log(bigramCount/unigram1Count/unigram2Count*sequenceLength*sequenceLength/(sequenceLength-1), 2)
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
			mergeCluster1Count = self.clusterNGramCounts[0].get(mergeCluster1, 0.0)
		if mergeCluster2Count is None:
			mergeCluster2Count = self.clusterNGramCounts[0].get(mergeCluster2, 0.0)
		if otherClusterCount is None:
			otherClusterCount = self.clusterNGramCounts[0].get(otherCluster, 0.0)
		# Consider, we could just combine the two bigram1 and two bigram2 counts as it won't change the mutual information equation. 
		# Update: No you can't!
		mutualInformation = 0.0
		bigram1Count = self.clusterNGramCounts[1].get((otherCluster, mergeCluster1),0.0)
		bigram2Count = self.clusterNGramCounts[1].get((otherCluster, mergeCluster2),0.0)
		mutualInformation += self.findMutualInformation(	(bigram1Count + bigram2Count),
															otherClusterCount,
															(mergeCluster1Count+mergeCluster2Count),
															sequenceLength
														)
		bigram1Count = self.clusterNGramCounts[1].get((mergeCluster1, otherCluster),0.0)
		bigram2Count = self.clusterNGramCounts[1].get((mergeCluster2, otherCluster),0.0)
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
			mergeCluster1Count = self.clusterNGramCounts[0].get(mergeCluster1, 0.0)
		if mergeCluster2Count is None:
			mergeCluster2Count = self.clusterNGramCounts[0].get(mergeCluster2, 0.0)
		bigram1Count = self.clusterNGramCounts[1].get((mergeCluster1, mergeCluster2),0.0)
		bigram2Count = self.clusterNGramCounts[1].get((mergeCluster2, mergeCluster1),0.0)
		bigram3Count = self.clusterNGramCounts[1].get((mergeCluster1, mergeCluster1),0.0)
		bigram4Count = self.clusterNGramCounts[1].get((mergeCluster2, mergeCluster2),0.0)
		totalBigramCount = bigram1Count + bigram2Count + bigram3Count + bigram4Count
		return self.findMutualInformation(	totalBigramCount,
											(mergeCluster1Count+mergeCluster2Count),
											(mergeCluster1Count+mergeCluster2Count), 
											sequenceLength
										)

	def findMutualInformationOfMergedClusters_PairPairConnection(	self, 
																	cluster1Group1, cluster2Group1,
																	cluster1Group2, cluster2Group2,
																	cluster1Group1Count=None, cluster2Group1Count=None, 
																	cluster1Group2Count=None, cluster2Group2Count=None, 
																	sequenceLength=None):
		raise NotImplementedError
		if mergeCluster1Count is None:
			mergeCluster1Count = self.clusterNGramCounts[0].get(mergeCluster1, 0.0)
		if mergeCluster2Count is None:
			mergeCluster2Count = self.clusterNGramCounts[0].get(mergeCluster2, 0.0)
		bigram1Count = self.clusterNGramCounts[1].get((mergeCluster1, mergeCluster2),0.0)
		bigram2Count = self.clusterNGramCounts[1].get((mergeCluster2, mergeCluster1),0.0)
		bigram3Count = self.clusterNGramCounts[1].get((mergeCluster1, mergeCluster1),0.0)
		bigram4Count = self.clusterNGramCounts[1].get((mergeCluster2, mergeCluster2),0.0)
		totalBigramCount = bigram1Count + bigram2Count + bigram3Count + bigram4Count
		return self.findMutualInformation(	totalBigramCount,
											(mergeCluster1Count+mergeCluster2Count),
											(mergeCluster1Count+mergeCluster2Count), 
											sequenceLength
										)

	# This function givest the merge reduction cost for a single cluster pair.
	# It is the cost of merging cluster 1 into cluster 2 
	# This algorithm has been verified. 
	def defineClusterMergeCostReductionTable_singleClusterPair(self, cluster1, cluster2, verbose=False):
		sequenceLength = self.sequenceLength
		# Consider: Should we store this separately? (i.e. clusters = self.clusters)
		clusters = self.clusterNGramCounts[0].keys()
		cluster1Count = self.clusterNGramCounts[0].get(cluster1, 0.0)
		cluster2Count = self.clusterNGramCounts[0].get(cluster2, 0.0)
		clusterCostReduction = 0.0
		clusterCostAddition = 0.0
		for cluster3 in clusters:
			if cluster3 == cluster1 or cluster3 == cluster2:
				continue #deal with these separately. 
			clusterCostReduction += self.getClusterCost(cluster1, cluster3)  # Encompasses 1->3 and 3->1
			clusterCostReduction += self.getClusterCost(cluster2, cluster3)  # Encompasses 2->3 and 3->2
			# This is the procedure you get if you try to combine two nodes into one:
			# P(c,c')*log(P(c,c')/P(c)/P(c'))
			cluster3Count = self.clusterNGramCounts[0].get(cluster3, 0.0)
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
		if not self.hasResetClusterCostTable:
			self.resetDataStatistics('defineClusterCostTable', verbose=verbose)
		start = timer()
		# Consider: Should we store this separately? (i.e. clusters = self.clusters)
		clusters = self.clusterNGramCounts[0].keys()
		numClusters = len(clusters)
		for i in range(numClusters):
			cluster1 = clusters[i]
			for j in range(i):
				cluster2 = clusters[j]
				mergeCost = self.defineClusterMergeCostReductionTable_singleClusterPair(cluster1, cluster2, verbose=verbose)
				self.clusterMergeCostReductionTable[self.getClusterCostBigram(cluster1, cluster2)] = mergeCost
		end = timer()
		self.hasResetClusterMergeCostReductionTable = True
		if verbose:
			print "Defined Cluster Merge Cost Reduction Table."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"


	def updateMergeCostReductionTable_UnmergedClusters(self, cluster1, cluster2, mergeCluster1, mergeCluster2, verbose=False):
		raise NotImplementedError
		cluster1Count = self.clusterNGramCounts[0].get(cluster1, 0.0)
		cluster2Count = self.clusterNGramCounts[0].get(cluster2, 0.0)
		mergeCluster1Count = self.clusterNGramCounts[0].get(mergeCluster1, 0.0)
		mergeCluster2Count = self.clusterNGramCounts[0].get(mergeCluster2, 0.0)
		currentMergeCost = self.clusterMergeCostReductionTable[self.getClusterCostBigram(cluster1, cluster2)] 
		mergeCostAddition = 0
		mergeCostAddition += self.getClusterCost(cluster1, mergeCluster1)
		mergeCostAddition += self.getClusterCost(cluster1, mergeCluster2)
		mergeCostAddition += self.getClusterCost(cluster2, mergeCluster1)
		mergeCostAddition += self.getClusterCost(cluster2, mergeCluster2)
		mergeCostReduction = 0
		mergeCostReduction += self.findMutualInformationOfMergedClusters_OtherConnection(mergeCluster1, mergeCluster2, cluster1) # m1 -> (1+2)
		mergeCostReduction += self.findMutualInformationOfMergedClusters_OtherConnection(mergeCluster1, mergeCluster2, cluster2) # m2 -> (1+2)

	def updateMergeCostReductionTable(self, mergeCluster1, mergeCluster2, verbose=False):
		raise NotImplementedError
		#TODO: We need to make sure we've updated clusterNGramCounts first
		# Consider: Should we store this separately? (i.e. clusters = self.clusters)
		clusters = self.clusterNGramCounts[0].keys()
		for i in range(numClusters):
			cluster1 = clusters[i]
			for j in range(i):
				cluster2 = clusters[j]
				if (cluster1 == mergeCluster1 or cluster1 == mergeCluster2 or cluster2 == mergeCluster1 or cluster2 == mergeCluster2):
					# TODO: Right?
					self.defineClusterMergeCostReductionTable_singleClusterPair(cluster1, cluster2, verbose=verbose)
				else:
					pass

	# TODO: Why are we passing in clusterNGram Counts? Why not self.clusterNGramCounts?
	#		Is it because we want to keep it generic for trying out new versions?
	def findClusteringCost(self, clusterNGramCounts, sequenceLength=None, verbose=False):
		start = timer()
		if sequenceLength is None:
			sequenceLength = self.sequenceLength
		# Consider: Should we store this separately? (i.e. clusters = self.clusters)
		clusters = clusterNGramCounts[0].keys()
		qualityCost = 0
		for cluster1 in clusters:
			cluster1Count = clusterNGramCounts[0].get(cluster1, 0)
			for cluster2 in clusters:
				cluster2Count = clusterNGramCounts[0].get(cluster2, 0)
				bigramCount = clusterNGramCounts[1].get((cluster1, cluster2),0)
				qualityCost += self.findMutualInformation(bigramCount, cluster1Count, cluster2Count)
		end = timer()
		if verbose:
			print "Found Clustering Cost."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"
		return qualityCost

	def findBrownClustering(self, numInitialClusters=100):
		raise NotImplementedError
		self.defineWordClusterMap(self, numInitialClusters=numInitialClusters)
		clusterSequence = self.defineClusterSequence()
		clusterNGramCounts = self.defineClusterCounts(clusterSequence)
		nextWordPointer = numInitialClusters
		unclusteredWords = self.NO_CLUSTER_SYMBOL in clusterNGramCounts[0]
		unmergedClusters = len(clusterNGramCounts) > 1
		while unclusteredWords or unmergedClusters:
			if unclusteredWords:
				changeWord = self.sortedWords[nextWordPointer]
				(clusterSequence, clusterNGramCounts) = self.changeWordCluster(	clusterSequence,
																				clusterNGramCounts,
																				changeWord,
																				nextWordPointer,
																				oldCluster=self.NO_CLUSTER_SYMBOL)
				nextWordPointer += 1
				unclusteredWords = self.NO_CLUSTER_SYMBOL in clusterNGramCounts[0]
			if unmergedClusters:
				raise NotImplementedError
				unmergedClusters = len(clusterNGramCounts) > 1


	def findMergeClusters(self, clusterSequence, clusterNGramCounts):
		raise NotImplementedError

	def changeCluster(self, clusterNGramCounts, clusterSequence, newCluster, oldCluster):
		raise NotImplementedError
		# clusterSequence = [self.START_CLUSTER_SYMBOL] + clusterSequence # TODO: will have to make sure it's a copy!
		sequenceLength = len(clusterSequence)
		# For Each Cluster
		for i in range(sequenceLength-1,0,-1):
			# For Each Cluster / If Cluster is Changing Cluster
			if clusterSequence[i] == oldCluster:
				# For Each Cluster / If Cluster is Changing Cluster / Adjust Unigram Counts
				clusterNGramCounts[0][newCluster] = clusterNGramCounts[0].get(newCluster,0) + 1
				clusterNGramCounts[0][oldCluster] -= 1
				if clusterNGramCounts[0][oldCluster] == 0:
					del(clusterNGramCounts[0][oldCluster]) #delete it so it throws an error if we try to access it later.			
				# For Each Cluster / If Cluster is Changing Cluster / Adjust Bigram Counts
				newBigram = (clusterSequence[i-1],newCluster)
				oldBigram = (clusterSequence[i-1],oldCluster)
				clusterNGramCounts[1][newBigram] = clusterNGramCounts[1].get(newBigram,0) + 1
				clusterNGramCounts[1][oldBigram] -= 1
				if clusterNGramCounts[1][oldBigram] == 0:
					del(clusterNGramCounts[1][oldBigram]) #delete it so it throws an error if we try to access it later.
				# For Each Cluster / If Cluster is Changing Cluster / Adjust Cluster Sequence
				clusterSequence[i] = newCluster	
		return (clusterSequence, clusterNGramCounts)

	def changeWordCluster(self, clusterNGramCounts, clusterSequence, changeWord, newCluster, oldCluster=None):
		raise NotImplementedError
		if oldCluster is None:
			oldCluster = self.wordClusterMapping[changeWord]
		# clusterSequence = [self.START_CLUSTER_SYMBOL] + clusterSequence
		sequenceLength = len(clusterSequence)
		# For Each Word
		for i in range(sequenceLength-1,0,-1):
			# For Each Word / If Word is Changing Word
			if self.wordSequence[i-1] == changeWord:
				# For Each Word / If Word is Changing Word / Adjust Unigram Counts
				clusterNGramCounts[0][newCluster] = clusterNGramCounts[0].get(newCluster,0) + 1
				clusterNGramCounts[0][oldCluster] -= 1
				if clusterNGramCounts[0][oldCluster] == 0:
					del(clusterNGramCounts[0][oldCluster]) #delete it so it throws an error if we try to access it later.			
				# For Each Word / If Word is Changing Word / Adjust Bigram Counts
				newBigram = (clusterSequence[i-1],newCluster)
				oldBigram = (clusterSequence[i-1],oldCluster)
				clusterNGramCounts[1][newBigram] = clusterNGramCounts[1].get(newBigram,0) + 1
				clusterNGramCounts[1][oldBigram] -= 1
				if clusterNGramCounts[1][oldBigram] == 0:
					del(clusterNGramCounts[1][oldBigram]) #delete it so it throws an error if we try to access it later.
				# For Each Word / If Word is Changing Word / Adjust Cluster Sequence
				clusterSequence[i] = newCluster
		return (clusterSequence, clusterNGramCounts)

####END TEST


# class ConditionalRandomFieldTagger:
# 	def __init__(self, trainingDataSet, indexColumn, dateColumn = None):
# 		self.trainingDataSet = trainingDataSet
# 		self.indexColumn = indexColumn
# 		self.dateColumn = dateColumn
# 		self.featureSet = []
# 		self.featureWeights = []

# 	def set_featureSet(self, featureSet):
# 		self.featureSet = featureSet
# 		self.featureWeights = np.zeros(shape=(featureSet.len,1))

# 	def calculateFeatures(featureSet,inDataSet):
# 		dataFeatures = np.zeros(shape=)

# 	def generate_featureSet(self):
# 		raise NotImplementedError


# #Very related to CRF Tagger... the same?
# class GreedyClusterAlgorithm:
# 	def __init__(self, dataSet, indexColumn, dateColumn = None):
# 		self.dataSet = dataSet
# 		self.indexColumn = indexColumn
# 		self.dateColumn = dateColumn
# 		self.featureSet = []

# 	def set_featureSet(self, featureSet):
# 		self.featureSet = featureSet

# 	def generateFeatures(self):
# 		for feature in self.featureSet:
# 			raise NotImplementedError
# 			#generate features 
# 			#generate signals from features 

# 	def cluster(self):
# 		for data in dataSet:
# 			raise NotImplementedError
# 			#use delta in features, applied on dataSet, to segment