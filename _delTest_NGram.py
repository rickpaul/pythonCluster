import pprint
import numpy as np
from clusterBrownian import NGramModel

if __name__ == '__main__':
	verbose = 0

	useRandSeed = 0
	seedOverride = None
	if not useRandSeed:
		seed = (np.random.randint(9999) if seedOverride is None else seedOverride)
		print 'Seed: ' + str(seed)
		np.random.seed(seed=seed)

	nGramLength = 5
	NGramMod = NGramModel(nGramLength)
	for i in range(4):
		data = np.random.randint(3, size = (1000,3))
		NGramMod.addTrainingData(data,verbose=verbose)

	NGramMod.establishMissingProbability(verbose=verbose)
	print NGramMod.findNGramProbability(2, NGramMod.tuplefy([[0,0,0],[0,0,1]]), tuple([0,0,0]))
	# print NGramMod.findSequenceProbability(np.array([[0,0,5],[0,0,0]]),True)
	# print NGramMod.findSequenceProbability(np.array([[0,0,1],[0,0,0]]),True)
	# print NGramMod.findSequenceProbability(np.array([[0,1,0],[0,0,0]]),True)
	for i in range(4):
		testLength = 1000.0
		sequence = np.random.randint(3, size = (testLength,3))
		perplexity = NGramMod.findSequenceProbability(sequence,padSequence=True,verbose=verbose)
		print str(perplexity/testLength)
	


