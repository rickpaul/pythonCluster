from math import floor
import matplotlib.pyplot as plt
import scipy.cluster.vq as spc
import numpy as np
from clusterMeanPrediction import ClusterMeanPredictionAlgorithm as CMPAlg

n = 24

offsets = [(1,1),(4,4),(1,3),(6,6),(9,3)]
x1 = np.random.random((n,2))-.5 + offsets[0]
x2 = np.random.random((n,2))-.5 + offsets[1]
x3 = np.random.random((n,2))-.5 + offsets[2]
x4 = np.random.random((n,2))-.5 + offsets[3]
x5 = np.random.random((n,2))-.5 + offsets[4]

x = np.concatenate((x1,x2,x3,x4,x5),axis=0)
idx = np.reshape(np.arange(0,len(x)),(len(x),1))

x = np.hstack((idx,x))

M = CMPAlg(x, 0)
####### Simple Flat Results Version
# print M.performMCCAlgorithm(1, 500, 5, 0.8)

####### Simple Distributions Version
# (Distribution, Periods) = M.performMCCAlgorithm(1, 500, 5, 0.8) 

# print Periods

# plt.hist(Distribution[:,2,0],bins = 30)
# plt.hist(Distribution[:,2,1],bins = 30)
# plt.show()

####### Complex Distributions Version
plt.hold(True)
startPeriod = 30
clusterStats = M.performCMPAlgorithm_Unweighted(startPeriod, 200, 5, .5) 
# print MarkovChainStats['statistics']
# print MarkovChainStats['statisticWeightsbyPeriod'] #Rewards small clusters
# print MarkovChainStats['statisticWeightsbyIteration']
# print MarkovChainStats['statisticWeightsbyPeriod'][:,0]
i = 1
periodsAhead = M.periodsAhead
periodAhead = 1
values1 = clusterStats['statistics'][:,periodAhead,0]
print "Bar " + str(i) + " : " + str(periodsAhead[periodAhead]) + " periods ahead. Expected is " + str(offsets[int(floor((startPeriod + periodsAhead[periodAhead])/n))][0])
i+=1
periodAhead = 4
print "Bar " + str(i) + " : " + str(periodsAhead[periodAhead]) + " periods ahead. Expected is " + str(offsets[int(floor((startPeriod + periodsAhead[periodAhead])/n))][0])
i+=1
values2 = clusterStats['statistics'][:,periodAhead,0]
periodAhead = 9
print "Bar " + str(i) + " : " + str(periodsAhead[periodAhead]) + " periods ahead. Expected is " + str(offsets[int(floor((startPeriod + periodsAhead[periodAhead])/n))][0])
values3 = clusterStats['statistics'][:,periodAhead,0]
i+=1

plt.hist(
	[values1, values2, values3],
	bins = 30, 
	)
plt.show()


# length = len(MarkovChainStats['statisticWeightsbyIteration'])
# weights = np.multiply(np.array(MarkovChainStats['statisticWeightsbyIteration']),np.reshape(MarkovChainStats['statisticWeightsbyPeriod'][:,periodAhead],(length,1)))
# weights = weights/np.mean(weights)
# values = np.reshape(MarkovChainStats['statistics'][:,periodAhead,0],(length,1))
# plt.hist(
# 	[values, values],
# 	bins = 30, 
# 	weights=[weights, np.ones((length,1))]
# 	)
# plt.show()