from timeit import default_timer as timer
import numpy as np
from copy import deepcopy
from math import log
import pudb # TEST
# import scipy.optimize.minimize as minimize
# import scipy.optimize.OptimizeResult as minResult

# See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
# See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult

class NGramModel:
	discount = .5
	nGrams = {}
	counts = {}
	missingProbabilityDensity = None
	missingProbabilityDensityCalibrated = False
	def __init__(self, nGramLength):
		self.nGramLength = nGramLength
		self.linInt0Max=1
		self.linInt1Max=3
		self.linInt2Max=5
		self.linearInterpolationWeights = np.ones(shape=(nGramLength,4))/nGramLength #These are the lambdas in linear interpolation

	def addTrainingData(self, data, padData=True, verbose=False):
		start = timer()
		# Flag Resetting
		self.missingProbabilityDensityCalibrated = False
		# Data Validation
		dataWidth = data.shape[1]
		if not hasattr(self, 'dataWidth'):
			self.dataWidth = dataWidth
		if dataWidth != self.dataWidth:
			# TODO: create bespoke dataCompatabilityError
			raise NotImplementedError
			#raise dataCompatabilityError
		# Data Padding
		if self.nGramLength > 1 and padData:
			data = np.vstack((np.array([[-1]*dataWidth]*(self.nGramLength-1)),data))
			data = np.vstack((data,np.array([[-2]*dataWidth])))
		dataLength = data.shape[0]
		# Create Statistics for All nGram Lengths
		for i in range(0,self.nGramLength):
			# Create Statistics for All nGram Lengths / Get Existing Statistics
			countsDict = self.counts.get(i,{})
			nGramsDict = self.nGrams.get(i,{})
			# Create Statistics for All nGram Lengths / Loop Through Training Data
			for j in range(self.nGramLength-1,dataLength):
				# Create Statistics for All nGram Lengths / Loop Through Training Data / Create Keys
				outerKey = self.tuplefy(data[j-i:j,:])
				innerKey = tuple(data[j])
				# Create Statistics for All nGram Lengths / Loop Through Training Data / Add Counts to Keys
				if outerKey not in nGramsDict:
					nGramsDict[outerKey] = {}
				nGramsDict[outerKey][innerKey] = nGramsDict[outerKey].get(innerKey,-1*self.discount) + 1
				countsDict[outerKey] = countsDict.get(outerKey,0) + 1
			# Create Statistics for All nGram Lengths / Save Statistics
			self.counts[i] = countsDict
			self.nGrams[i] = nGramsDict
		end = timer()
		if verbose:
			print "Added Sentence of length " + str(dataLength - self.nGramLength - 1) + " to data model."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	def establishMissingProbability(self, verbose=False):
		start = timer()
		self.missingProbabilityDensity = deepcopy(self.counts)
		for nGram, outerKeys in self.nGrams.iteritems():
			for outerKey, innerKeys in outerKeys.iteritems():
				totalCount = self.counts[nGram][outerKey]
				missingDensity = len(innerKeys) * self.discount
				self.missingProbabilityDensity[nGram][outerKey] = missingDensity / totalCount
		self.missingProbabilityDensityCalibrated = True
		end = timer()
		if verbose:
			print "Recalibrated Missing Word Probability Density."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	def findSequenceProbability(self, sequence, padSequence=True, verbose=False):
		if not self.missingProbabilityDensityCalibrated:
			self.establishMissingProbability()
		start = timer()
		if self.counts[0][()] == 0:
			# TODO: create bespoke notEnoughDataError
			raise NotImplementedError
			#raise notEnoughDataError
		sequenceWidth = sequence.shape[1]
		if sequenceWidth != self.dataWidth:
			# TODO: create bespoke dataCompatabilityError
			raise NotImplementedError
			#raise dataCompatabilityError
		origSequenceLength = sequence.shape[0]
		if self.nGramLength > 1 and padSequence:
			sequence = np.vstack((np.array([[-1]*sequenceWidth]*(self.nGramLength-1)),sequence))
			sequence = np.vstack((sequence,np.array([[-2]*sequenceWidth])))
		sequenceLength = sequence.shape[0]
		logProbability = 0
		for j in range(self.nGramLength-1,sequenceLength):
			if verbose:
				print '\t' + str(self.tuplefy(sequence[j-self.nGramLength+1:j])) + '->' + str(tuple(sequence[j:j+1][0]))
			seqProbability = 0
			for i in range(0, self.nGramLength):
				prevSequence = self.tuplefy(sequence[j-i:j,:])
				currentWord = tuple(sequence[j:j+1][0])
				addProb = self.findNGramProbability(i, prevSequence, currentWord,depth=0)
				if prevSequence not in self.counts[i]:
					addProb *= self.linearInterpolationWeights[i][0]
				elif self.counts[i][prevSequence] > 5:
					addProb *= self.linearInterpolationWeights[i][3]
				elif self.counts[i][prevSequence] > 2:
					addProb *= self.linearInterpolationWeights[i][2]
				elif self.counts[i][prevSequence] > 0:
					addProb *= self.linearInterpolationWeights[i][1]
				else:
					addProb *= self.linearInterpolationWeights[i][0]
				if verbose:
					print '\t\t' + str(prevSequence) + '->' + str(currentWord) + ' | ' + str(round(log(addProb,2)*100000)/100000.0)
				seqProbability += addProb
			logProbability = logProbability + log(seqProbability,2)
			if verbose:
				print '\tSequence Probability : ' + str(round(log(seqProbability,2)*100000)/100000.0)
		if verbose:
				print 'Total Probability : ' + str(round(logProbability*100000)/100000.0)
		end = timer()
		if verbose:
			print "Found sequence probability for sequence of length " + str(origSequenceLength) + "."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"
		return logProbability

	# Recursive Implementation of Katz Backoff Model
	def findNGramProbability(self, nGram, prevSequence, currentWord,depth=1):
		# If the current word has never been seen before,
		# return the probability as 1/Vocabulary
		if nGram == -1:
			# TODO: Verify this is appropriate.
			return 1.0 / len(self.nGrams[0][()])
		# If the prefix sequence word has never been seen before,
		# return the probability of an unseen random sequence * the chance of the current word
		if prevSequence not in self.nGrams[nGram]:
			# TODO: Verify this is appropriate
			return 1.0 / len(self.nGrams[nGram]) * self.findNGramProbability(0, (), currentWord)
		# If the prefix sequence word has been seen before and the current word has been seen before,
		# return the probability as calculated
		if currentWord in self.nGrams[nGram][prevSequence]:
			return self.nGrams[nGram][prevSequence][currentWord] / self.counts[nGram][prevSequence]
		# If the prefix sequence word has been seen before but the current word has never been seen before, 
		# return the adjusted probability using missing probility data
		else:
			normalizingProbability = 1
			for word in self.nGrams[nGram][prevSequence].keys():
				probRed = self.findNGramProbability(nGram - 1, prevSequence[1:len(prevSequence)], word)
				normalizingProbability -= probRed
			return 	self.missingProbabilityDensity[nGram][prevSequence] * \
					self.findNGramProbability(nGram - 1, prevSequence[1:len(prevSequence)], currentWord) * \
					(1.0 / normalizingProbability)

	def tuplefy(self, array2D):
		tupleList = []
		for subArray in array2D:
			tupleList.append(tuple(subArray))
		return tuple(tupleList)

# class BrownClusterAlgorithm:
# 	def __init__(self, dataSet, indexColumn, dateColumn = None):
# 		self.dataSet = dataSet
# 		self.indexColumn = indexColumn
# 		self.dateColumn = dateColumn

# class HiddenMarkovModelTagger:
# 	def __init__(self, trainingDataSet, labelColumn, indexColumn):
# 		self.trainingDataSet = trainingDataSet
# 		self.trainingLabels = trainingDataSet[:,labelColumn]
# 		self.indexColumn = indexColumn
# 		self.labelColumn = labelColumn

# 	def perform_ViterbiAlgorithm(self):
# 		raise NotImplementedError
# 	def perform_ForwardBackwardAlgorithm(self):
# 		raise NotImplementedError


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