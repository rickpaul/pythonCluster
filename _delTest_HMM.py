from timeit import default_timer as timer
import pprint
import numpy as np
from copy import deepcopy
from math import log
from clusterBrownian import DataCompatabilityError
from clusterBrownian import NoDataError

class HMMModel:
	tagWordEmissionCounts = {} 		# chance of a word given a tag (i.e. the emission probability)
	singleTagCounts = {}
	# The singleTagCounts can be found from summing over the tagWordEmissionCounts for a tag, then adjusting for discount. 
	# But hopefully this is faster.
	singleWordCounts = {}
	tagNGrams = {}
	tagCounts = {}
	# missingProbabilityDensity = None
	# missingProbabilityDensityCalibrated = False
	def __init__(self, nGramLength, multipleLengthNGram=True, nGramDiscount=.5, tagWordDiscount=.5):
		self.totalWordsAdded = 0
		self.nGramDiscount = nGramDiscount
		self.tagWordDiscount = tagWordDiscount
		self.nGramLength = nGramLength
		self.multipleLengthNGram = multipleLengthNGram
		self.linearInterpolationWeights = np.ones(shape=(nGramLength,4))/nGramLength #These are the lambdas in linear interpolation

	def addTrainingData(self, wordData, tagData, padData=True, verbose=False):
		start = timer()
		# Flag Resetting
		self.missingProbabilityDensityCalibrated = False
		# Data Validation
		# Data Validation / Word Data Width
		wordDataWidth = wordData.shape[1]
		if not hasattr(self, 'wordDataWidth'):
			self.wordDataWidth = wordDataWidth		
		if wordDataWidth != self.wordDataWidth:
			raise DataCompatabilityError('wordDataWidth', wordDataWidth, self.wordDataWidth)
		# Data Validation / Tag Data Width
		tagDataWidth = tagData.shape[1]
		if not hasattr(self, 'tagDataWidth'):
			self.tagDataWidth = tagDataWidth
		if tagDataWidth != self.tagDataWidth:
			raise DataCompatabilityError('tagDataWidth', tagDataWidth, self.tagDataWidth)
		# Data Validation / Tag Data Length = Word Data Length
		wordDataLength = wordData.shape[0]
		tagDataLength = tagData.shape[0]
		if tagDataLength != wordDataLength:
			raise DataCompatabilityError('tagDataLength', tagDataLength, wordDataLength)

		# Create Statistics for Tag/Word Emissions and Tag Probabilities
		for j in range(self.nGramLength-1,tagDataLength):
			tagKey = tuple(tagData[j])
			self.singleTagCounts[tagKey] = self.singleTagCounts.get(tagKey,0) + 1
			wordKey = tuple(wordData[j])
			self.singleWordCounts[wordKey] = self.singleTagCounts.get(tagKey,0) + 1
			if tagKey not in self.tagWordEmissionCounts:
				self.tagWordEmissionCounts[tagKey] = {}
			self.tagWordEmissionCounts[tagKey][wordKey] = self.tagWordEmissionCounts[tagKey].get(wordKey,-1*self.tagWordDiscount) + 1 # CONSIDER: Do we want to discount it?
			
		# Data Padding
		if self.nGramLength > 1 and padData:
			tagData = np.vstack((np.array([[-1]*tagDataWidth]*(self.nGramLength-1)),tagData))
			tagData = np.vstack((tagData,np.array([[-2]*tagDataWidth])))
		tagDataLength = tagData.shape[0]
		# Decide Whether to Have Multiple nGram Lengths
		minNGram = 0 if self.multipleLengthNGram else self.nGramLength-1
		# Create Statistics for All Tag nGram Lengths
		for i in range(minNGram,self.nGramLength):
			newTagCounts = self.tagCounts.get(i,{})
			newTagNGrams = self.tagNGrams.get(i,{})
			# Create Statistics for All nGram Lengths / Loop Through Training Data
			for j in range(self.nGramLength-1,tagDataLength):
				# Create Statistics for All nGram Lengths / Loop Through Training Data / Create Tag Keys
				outerKey = self.tuplefy(tagData[j-i:j,:])
				innerKey = tuple(tagData[j])
				# Create Statistics for All nGram Lengths / Loop Through Training Data / Add Counts to tag Keys
				if outerKey not in newTagNGrams:
					newTagNGrams[outerKey] = {}
				newTagNGrams[outerKey][innerKey] = newTagNGrams[outerKey].get(innerKey,-1*self.nGramDiscount) + 1
				newTagCounts[outerKey] = newTagCounts.get(outerKey,0) + 1
			self.tagCounts[i] = newTagCounts
			self.tagNGrams[i] = newTagNGrams
		self.totalWordsAdded += wordDataLength
		end = timer()
		if verbose:
			print "Added Sentence of length " + str(wordDataLength) + " to data model."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	# CONSIDER: This is the same as the NGramModel version, but with different variable names
	def establishMissingProbability(self, verbose=False):
		start = timer()
		self.missingProbabilityDensity = deepcopy(self.tagCounts)
		for nGram, outerKeys in self.tagNGrams.iteritems():
			for outerKey, innerKeys in outerKeys.iteritems():
				totalCount = self.tagCounts[nGram][outerKey]
				missingDensity = len(innerKeys) * self.nGramDiscount
				self.missingProbabilityDensity[nGram][outerKey] = missingDensity / totalCount
		self.missingProbabilityDensityCalibrated = True
		end = timer()
		if verbose:
			print "Recalibrated Missing Word Probability Density."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"

	# This is a naive method of finding the probability of a sentence. We won't dwell on its workings too much.
	def findSequenceProbability(self, wordSequence, tagSequence, padSequence=True, verbose=False):
		if not self.missingProbabilityDensityCalibrated:
			self.establishMissingProbability()
		start = timer()
		# Data Validation
		# Data Validation / Have We Added Data?
		if len(self.tagCounts) == 0:
			raise NoDataError
		# Data Validation / Word Data Width
		wordDataWidth = wordSequence.shape[1]
		if wordDataWidth != self.wordDataWidth:
			raise DataCompatabilityError('wordDataWidth', wordDataWidth, self.wordDataWidth)
		# Data Validation / Tag Data Width
		tagDataWidth = tagSequence.shape[1]
		if tagDataWidth != self.tagDataWidth:
			raise DataCompatabilityError('tagDataWidth', tagDataWidth, self.tagDataWidth)
		# Data Validation / Tag Data Length = Word Data Length
		wordDataLength = wordSequence.shape[0]
		tagDataLength = tagSequence.shape[0]
		if tagDataLength != wordDataLength:
			raise DataCompatabilityError('tagDataLength', tagDataLength, self.tagDataLength)
		# Data Padding
		origSequenceLength = wordSequence.shape[0]
		if self.nGramLength > 1 and padSequence:
			tagSequence = np.vstack((np.array([[-1]*tagDataWidth]*(self.nGramLength-1)),tagSequence))
			tagSequence = np.vstack((tagSequence,np.array([[-2]*tagDataWidth])))
		tagSequenceLength = tagSequence.shape[0]
		logProbability = 0
		for j in range(self.nGramLength-1,tagSequenceLength):
			if verbose:
				print '\t' + str(self.tuplefy(tagSequence[j-self.nGramLength+1:j])) + '->' + str(tuple(tagSequence[j:j+1][0]))
			seqProbability = 0
			for i in range(0, self.nGramLength):
				prevTagSequence = self.tuplefy(tagSequence[j-i:j,:])
				currentTag = tuple(tagSequence[j])
				addProb = self.findNGramProbability(i, prevTagSequence, currentTag,depth=0)
				if False:
					print '\t\t' + str(prevTagSequence) + '->' + str(currentTag) + ' | ' + str(round(log(addProb,2)*100000)/100000.0)
				seqProbability += addProb
			logProbability = logProbability + log(seqProbability,2)
			if verbose:
				print '\tSequence Probability : ' + str(round(log(seqProbability,2)*100000)/100000.0)
			emissionProbability = findSingleWordEmissionProbability(currentTag, tuple(wordSequence[j]))
		if verbose:
				print 'Total Probability : ' + str(round(logProbability*100000)/100000.0)
		end = timer()
		if verbose:
			print "Found sequence probability for sequence of length " + str(origSequenceLength) + "."
			print "Time elapsed: " + str(round((end-start)*1000)/1000.0) + "s"
		return logProbability

	def findSingleWordEmissionProbability(self, currentTag, currentWord, depth=1):
		if currentTag not in self.tagWordEmissionCounts:
			# This shouldn't really happen in the NLP world, as they're supplying tags.
			# Where I'm generating tags, though, this does need to be taken into account.
			# TODO: Verify if this is correct.
			if currentWord in self.singleWordCounts:
				return self.singleWordCounts[currentWord]/self.totalWordsAdded
			else:
				return 1.0/self.totalWordsAdded
		elif currentWord not in self.tagWordEmissionCounts[currentTag]:
			# TODO: Verify if this is correct.
			if currentWord in self.singleWordCounts:
				return self.missingProbabilityDensity[1][(currentTag,)] * self.singleWordCounts[currentWord]/self.totalWordsAdded
			else:
				return self.missingProbabilityDensity[1][(currentTag,)] * 1.0/self.totalWordsAdded			
		else:
			return self.tagWordEmissionCounts[currentTag][currentWord]/self.singleTagCounts[currentTag]

	# Recursive Implementation of Katz Backoff Model
	# CONSIDER: This is the same as the NGramModel version, but with different variable names
	def findNGramProbability(self, nGram, prevSequence, currentWord, depth=1):
		# If the current word has never been seen before,
		# return the probability as 1/Vocabulary
		if nGram == -1:
			# TODO: Verify this is appropriate.
			return 1.0 / (len(self.tagNGrams[0][()]) + 1)
		# If the prefix sequence word has never been seen before,
		# return the probability of an unseen random sequence * the chance of the current word
		if prevSequence not in self.tagNGrams[nGram]:
			# TODO: This is not correct. 
			# 		You can show that it's not ok by summing over probabilities for impossible sequences.
			# 		Those probabilities should still add to 1
			# return 1.0 / len(self.nGrams[nGram]) * self.findNGramProbability(0, (), currentWord)
			# TODO: Verify this fix. It's kind of cheating because we're double-counting previous sequences.
			return self.findNGramProbability(nGram - 1, prevSequence[1:len(prevSequence)], currentWord)
		# If the prefix sequence word has been seen before and the current word has been seen before,
		# return the probability as calculated
		if currentWord in self.tagNGrams[nGram][prevSequence]:
			return self.tagNGrams[nGram][prevSequence][currentWord] / self.tagCounts[nGram][prevSequence]
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
			for word in self.tagNGrams[nGram][prevSequence].keys():
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


if __name__ == '__main__':
	verbose = 1

	useRandSeed = 0
	seedOverride = 737
	if not useRandSeed:
		seed = (np.random.randint(9999) if seedOverride is None else seedOverride)
		print 'Seed: ' + str(seed)
		np.random.seed(seed=seed)

	nGramLength = 3
	HMMMod = HMMModel(nGramLength)
	for i in range(4):
		dataLen=1000
		wordData = np.random.randint(3, size = (dataLen,3))
		tagData = np.vstack((np.sum(wordData,1),np.random.randint(10, size = (dataLen)))).T
		HMMMod.addTrainingData(wordData, tagData,verbose=verbose)

	HMMMod.establishMissingProbability(verbose=verbose)
	# Test findSingleWordEmissionProbability #################
	total = 0
	for i in range(10):
		p = HMMMod.findSingleWordEmissionProbability((1,i), (0,0,1))
		total += p 
		print i,
		print p
	print total

	# Test findSequenceProbability #################
	# Test findNGramProbability #################
	# total = 0
	# for i in range(8):
	# 	print i,
	# 	p = HMMMod.findNGramProbability(2, ((3,0),(3,0)), (i,1))
	# 	p2 = HMMMod.findNGramProbability(2, ((3,0),(3,0)), (i,0))
	# 	total += (p + p2)  
	# 	print str(p) + '\t',
	# 	print p2
	# print total
	# for i in range(4):
	# 	testLength = 100.0
	# 	data = np.random.randint(3, size = (testLength,3))
	# 	perplexity = HMMMod.findSequenceProbability(data,verbose=verbose)
	# 	print str(perplexity/testLength)
	



