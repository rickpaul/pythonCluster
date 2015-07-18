from timeit import default_timer as timer
import numpy as np
from copy import deepcopy
from math import log


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

class NGramModel:
	nGrams = {}
	counts = {}
	missingProbabilityDensity = None
	missingProbabilityDensityCalibrated = False
	def __init__(self, nGramLength,discount=.5):
		# TODO: Different discounts for different nGramLengths?
		# 		In general, discount should be found variably, by finding the average level of conviction 
		#		(the avg. value of inner key values in the missingProbDensity algorithm). If we're convinced,
		#		lower discount. It should also be a function of how much test data we can provide.
		self.discount = discount
		self.nGramLength = nGramLength
		self.linInt0Max=1 # Deprecated? We just use the numbers below.
		self.linInt1Max=3 # Deprecated? We just use the numbers below.
		self.linInt2Max=5 # Deprecated? We just use the numbers below.
		self.linearInterpolationWeights = np.ones(shape=(nGramLength,4))/nGramLength #These are the lambdas in linear interpolation

	def addTrainingData(self, data, padData=True, verbose=False):
		start = timer()
		# Flag Resetting
		self.missingProbabilityDensityCalibrated = False
		# Data Validation
		# Data Validation / Word Data Width
		dataWidth = data.shape[1]
		if not hasattr(self, 'dataWidth'):
			self.dataWidth = dataWidth
		if dataWidth != self.dataWidth:
			raise DataCompatabilityError('dataWidth', dataWidth, self.dataWidth)
		# Data Padding
		if self.nGramLength > 1 and padData:
			data = np.vstack((np.array([[-1]*dataWidth]*(self.nGramLength-1)),data))
			data = np.vstack((data,np.array([[-2]*dataWidth])))
		dataLength = data.shape[0]
		# Create Statistics for All nGram Lengths
		for i in range(0,self.nGramLength):
			# Create Statistics for All nGram Lengths / Get Existing Statistics
			newCountsDict = self.counts.get(i,{})
			newNGramsDict = self.nGrams.get(i,{})
			# Create Statistics for All nGram Lengths / Loop Through Training Data
			for j in range(self.nGramLength-1,dataLength):
				# Create Statistics for All nGram Lengths / Loop Through Training Data / Create Keys
				outerKey = self.tuplefy(data[j-i:j,:])
				innerKey = tuple(data[j])
				# Create Statistics for All nGram Lengths / Loop Through Training Data / Add Counts to Keys
				if outerKey not in newNGramsDict:
					newNGramsDict[outerKey] = {}
				newNGramsDict[outerKey][innerKey] = newNGramsDict[outerKey].get(innerKey,-1*self.discount) + 1
				newCountsDict[outerKey] = newCountsDict.get(outerKey,0) + 1
			# Create Statistics for All nGram Lengths / Save Statistics
			self.counts[i] = newCountsDict
			self.nGrams[i] = newNGramsDict
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
		# Data Validation
		# Data Validation / Have We Added Data?
		if len(self.counts) == 0:
			raise NoDataError
		# Data Validation / Word Data Width
		sequenceWidth = sequence.shape[1]
		if sequenceWidth != self.dataWidth:
			raise DataCompatabilityError('dataWidth', sequenceWidth, self.dataWidth)
		origSequenceLength = sequence.shape[0]
		# Data Padding
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
				currentWord = tuple(sequence[j])
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
	# CONSIDER: depth was used for debugging, consider removing.
	def findNGramProbability(self, nGram, prevSequence, currentWord, depth=1): 
		# If the current word has never been seen before,
		# return the probability as 1/Vocabulary
		if nGram == -1:
			# TODO: Fix this. It gives too high odds for small vocabularies.
			# 		It should be something like the odds we've never seen this sequence before. 
			#		(Right now, we're just taking a degrees of freedom approach)
			return 1.0 / (len(self.nGrams[0][()]) + 1)
		# If the prefix sequence word has never been seen before,
		# return the probability of an unseen random sequence * the chance of the current word
		if prevSequence not in self.nGrams[nGram]:
			# TODO: This is not correct. 
			# 		You can show that it's not ok by summing over probabilities for impossible sequences.
			# 		Those probabilities should still add to 1
			# return 1.0 / len(self.nGrams[nGram]) * self.findNGramProbability(0, (), currentWord)
			# TODO: Verify this fix. It's kind of cheating because we're double-counting previous sequences.
			return self.findNGramProbability(nGram - 1, prevSequence[1:len(prevSequence)], currentWord)
		# If the prefix sequence word has been seen before and the current word has been seen before,
		# return the probability as calculated
		if currentWord in self.nGrams[nGram][prevSequence]:
			return self.nGrams[nGram][prevSequence][currentWord] / self.counts[nGram][prevSequence]
		# If the prefix sequence word has been seen before but the current word has never been seen before, 
		# return the adjusted probability using missing probility data
		else:
			# It is not obvious, but was confirmed with some effort, that this is a better way to find the 
			# normalizing probability statistic. If you do it the suggested way, by counting all potential
			# words and summing them up, it is workable, but mUch slower, and needs to be adjusted for the
			# missing probability density. This way avoids both of those, and is cleaner.
			# Note. I am still not sure that the methodology of the whole fn is correct (it produces funky
			# perpexities for nGramLength>1); I am just sure that this method is the same as counting up.
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
