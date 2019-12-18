
# standard modules
import matplotlib as mpl
import numpy as np
import matplotlib.pylab as plt
import h5py
import os
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr
from prediction import userTracker
#
mpl.rcParams['interactive']  = False
###############################################    
# 
#    run parameters
#
###############################################
typ = 'AML32' # possible values AML32, AML18, AML70
condition = 'moving'# # Moving, immobilized, chip
first = True # if true, create new HDF5 file
###############################################    
# 
#    load data into dictionary
#
##############################################
path = userTracker.dataPath()
folder = os.path.join(path, '{}_{}/'.format(typ, condition))
dataLog = os.path.join(path,'{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
outLocData = os.path.join(path,'/Analysis/{}_{}.hdf5'.format(typ, condition))

# data parameters
dataPars = {'medianWindow':50, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':100, # gauss window for angle velocity derivative. Acts on full (50Hz) data
            'rotate':False, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 6,  # gauss window for red and green channel
            'interpolateNans': 6,#interpolate gaps smaller than this of nan values in calcium data
            }


dataSets = dh.loadMultipleDatasets(dataLog, pathTemplate=folder, 
                                   dataPars = dataPars, nDatasets = 5)
keyList = np.sort(dataSets.keys())
    
print keyList
keyList = keyList[:]
# results dictionary 
resultDict = {}
for kindex, key in enumerate(keyList):
    resultDict[key] = {}
# analysis parameters

pars ={'nCompPCA':10, # no of PCA components
        'PCAtimewarp':False, #timewarp so behaviors are equally represented
        'trainingCut': 0.6, # what fraction of data to use for training 
        'trainingType': 'middle', # simple, random or middle.select random or consecutive data for training. Middle is a testset in the middle
        'linReg': 'simple', # ordinary or ransac least squares
        'trainingSample': 1, # take only samples that are at least n apart to have independence. 4sec = gcamp_=->24 apart
        'useRank': 0, # use the rank transformed version of neural data for all analyses
        'useDeconv': 0, # use the deconvolved transformed version of neural data for all analyses
        'nCluster': 10, # use the deconvolved transformed version of neural data for all analyses
        'useClust':False,# use clusters in the fitting procedure.
        'useDeriv':False,# use neural activity derivative for PCA
        'useRaw':False,# use neural R/R0
        'testVolumes' : 6*60*1, # 2 min of data for test sets in nested validation
        'periods': np.arange(0, 300) # relevant periods in seconds for timescale estimate
        
     }

behaviors = ['AngleVelocity','Eigenworm3']
#behaviors = ['Eigenworm3']

###############################################    
# 
# check which calculations to perform
#
##############################################
createIndicesTest = 1#True 
overview = 1#False
predNeur = 0
predPCA = 0
bta = 0
svm = 0
pca = 1#False
kato_pca= 0
half_pca= 0
hierclust = False
linreg = False
periodogram = 0
nestedvalidation = 0
lasso = 0
elasticnet = 0#True
lagregression = 0
positionweights = 0#True
resultsPredictionOverview = 0
transient = 0
###############################################    
# 
# create training and test set indices
# 
##############################################
if createIndicesTest:
    for kindex, key in enumerate(keyList):
        resultDict[key] = {'Training':{}}
        for label in behaviors:
            train, test = dr.createTrainingTestIndices(dataSets[key], pars, label=label)
            if transient:
               train = np.where(dataSets[key]['Neurons']['Time']<4*60)[0]
                # after 4:30 min
               test = np.where((dataSets[key]['Neurons']['Time']>7*60)*(dataSets[key]['Neurons']['Time']<14*60))[0]
               resultDict[key]['Training']['Half'] ={'Train':train}
               resultDict[key]['Training']['Half']['Test'] = test
            else:
                 # add half split
                midpoint = np.mean(dataSets[key]['Neurons']['Time'])
                trainhalf = np.where(dataSets[key]['Neurons']['Time']<midpoint)[0]
                testhalf = np.where(dataSets[key]['Neurons']['Time']>midpoint)[0]
                resultDict[key]['Training']['Half'] ={'Train':trainhalf}
                resultDict[key]['Training']['Half']['Test'] = testhalf
            resultDict[key]['Training'][label] = {'Train':train  }
            resultDict[key]['Training'][label]['Test']=test
           

    print "Done generating trainingsets"

###############################################    
# 
# some generic data checking plots
#
##############################################
if overview:
        # line plots of neuronal activity, pretty
    mp.neuralActivity(dataSets, keyList)
        # cimple scatter of behavior versus neurons
    #mp.plotBehaviorNeuronCorrs(dataSets, keyList, behaviors)
        # heatmaps of neuronal activity ordered by behavior
    #mp.plotBehaviorOrderedNeurons(dataSets, keyList, behaviors)
        # sanity check - CMS velocity, wave velocity and turn variables with ethogram
    #mp.plotVelocityTurns(dataSets, keyList)
        # plot neural data, ethogram, behavior and location for each dataset
    mp.plotDataOverview(dataSets, keyList)
        # neuron locations
    #mp.plotNeurons3D(dataSets, keyList, threed = False)  
    #mp.plotExampleCenterlines(dataSets, keyList, folder)
    plt.show() 
    
###############################################    
# 
# predict neural dynamics from behavior
#
##############################################
if predPCA:
    for kindex, key in enumerate(keyList):
        print 'predicting neural dynamics from behavior'
        splits = resultDict[key]['Training']
        resultDict[key]['PCAPred'] = dr.predictBehaviorFromPCA(dataSets[key], \
                    splits, pars, behaviors)
    
###############################################    
# 
# predict neural dynamics from behavior
#
##############################################
if predNeur:
    for kindex, key in enumerate(keyList):
        print 'predicting neural dynamics from behavior'
        splits = resultDict[key]['Training']
        resultDict[key]['RevPred'] = dr.predictNeuralDynamicsfromBehavior(dataSets[key], \
                    splits, pars, useFullNeurons=False)
        mp.plotNeuronPredictedFromBehavior(resultDict[key], dataSets[key])
        plt.show()
###############################################    
# 
# use agglomerative clustering to connect similar neurons
#
##############################################
if hierclust:
    for kindex, key in enumerate(keyList):
        print 'running clustering'
        resultDict[key]['clust'] = dr.runHierarchicalClustering(dataSets[key], pars)
        
###############################################    
# 
# calculate the periodogram of the neural signals
#
##############################################
if periodogram:
    print 'running periodogram(s)'
    for kindex, key in enumerate(keyList):
        resultDict[key]['Period'] = dr.runPeriodogram(dataSets[key], pars, testset = None)       
###############################################    
# 
# use behavior triggered averaging to create non-othogonal axes
#
##############################################
if bta:
    for kindex, key in enumerate(keyList):
        print 'running BTA'
        resultDict[key]['BTA'] =dr.runBehaviorTriggeredAverage(dataSets[key], pars)
    mp.plotPCAresults(dataSets, resultDict, keyList, pars,  flag = 'BTA')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho', flag = 'BTA')
    plt.show()
    ###############################################    
# 
# use svm to predict discrete behaviors
#
##############################################
if svm:
    for kindex, key in enumerate(keyList):
        print 'running SVM'
        splits = resultDict[key]['Training']
        resultDict[key]['SVM'] = dr.discreteBehaviorPrediction(dataSets[key], pars, splits )
    
    
    # overview of SVM results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList, pars,  flag = 'SVM')
    plt.show()
    #  plot 3D trajectory of SVM
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho', flag = 'SVM')
    plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'time',  flag = 'SVM')
#        plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'velocity',  flag = 'SVM')
#        plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'turns',  flag = 'SVM')
#        plt.show()

###############################################    
# 
# run PCA and store results
#
##############################################
#%%
if pca:
    print 'running PCA'
    for kindex, key in enumerate(keyList):
        resultDict[key]['PCA'] = dr.runPCANormal(dataSets[key], pars)

    # overview of data ordered by PCA
    mp.plotDataOverview2(dataSets, keyList, resultDict)
    # overview of PCA results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList, pars)
    plt.show()
    
    mp.plotPCANoise(resultDict, keyList)

    # show correlates of PCA
    #mp.plotPCAcorrelates(dataSets, resultDict, keyList, pars, flag='PCA')
    #  plot 3D trajectory of PCA
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho')
#    plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'time')
#        plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'velocity')
#        plt.show()
#        mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'turns')
#        plt.show()
###############################################    
# 
# run Kato PCA
#
##############################################
#%%
if kato_pca:
    print 'running Kato et. al PCA'
    for kindex, key in enumerate(keyList):
        resultDict[key]['katoPCA'] = dr.runPCANormal(dataSets[key], pars, deriv = True)
    
    # overview of PCA results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList, pars, flag='katoPCA')
    plt.show()
    
   
    # show correlates of PCA
    mp.plotPCAcorrelates(dataSets, resultDict, keyList, pars, flag='katoPCA')
    #  plot 3D trajectory of PCA
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho', flag='katoPCA')
    plt.show()
###############################################    
# 
# run split first-second half PCA
#
##############################################
#%%
if half_pca:
    print 'half-split PCA'
    for kindex, key in enumerate(keyList):
        # run PCA on each half
        splits = resultDict[key]['Training']
        
        resultDict[key]['PCAHalf1'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset = splits['Half']['Train'])
        resultDict[key]['PCAHalf2'] = dr.runPCANormal(dataSets[key], pars, whichPC=0, testset =splits['Half']['Test'])
        resultDict[key]['PCArankCorr'] = dr.rankCorrPCA(resultDict[key])
    
################################################    
## 
## estimate noise level pca shuffle
##
###############################################
##%%
#print 'estimate PCA noise level'
#if pca_noise:
#    for kindex, key in enumerate(keyList):
#        # run PCA on each half
#        splits = resultDict[key]['Training']
#        resultDict[key]['PCANoise'] = dr.runPCANoiseLevelEstimate(dataSets[key], pars)
#        resultDict[key]['PCAHalf1Noise'] = dr.runPCANoiseLevelEstimate(dataSets[key], pars, testset = splits['Half']['Train'])
#        resultDict[key]['PCAHalf2Noise'] = dr.runPCANoiseLevelEstimate(dataSets[key], pars,  testset =splits['Half']['Test'])
#    mp.plotPCANoise(resultDict, keyList)
#%%
###############################################    
# 
# linear regression single neurons
#
##############################################
if linreg:
    for kindex, key in enumerate(keyList):
        splits = resultDict[key]['Training']
        resultDict[key]['Linear Regression'] = dr.linearRegressionSingleNeuron(dataSets[key], pars, splits)
    
    mp.plotLinearPredictionSingleNeurons(dataSets, resultDict, keyList)
    plt.show()

#%%
###############################################    
# 
# linear regression using LASSO
#
##############################################
if lasso:
    print "Performing LASSO.",
    for kindex, key in enumerate(keyList):
        print key
        splits = resultDict[key]['Training']
        resultDict[key]['LASSO'] = dr.runLasso(dataSets[key], pars, splits, plot=1, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key],splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
        for tmpKey in tmpDict.keys():
            resultDict[key]['LASSO'][tmpKey].update(tmpDict[tmpKey])
#            # reorganize to get similar structure as PCA
#            tmpDict = dr.reorganizeLinModel(dataSets[key], resultDict[key], splits, pars, fitmethod = 'LASSO', behaviors = behaviors)
#            for tmpKey in tmpDict.keys():
#                resultDict[key]['LASSO'][tmpKey]=tmpDict[tmpKey]
#            
#        # do converse calculation -- give it only the neurons non-zero in previous case
#        subset = {}
        subset['AngleVelocity'] = np.where(resultDict[key]['LASSO']['Eigenworm3']['weights']>0)[0]
        subset['Eigenworm3'] = np.where(resultDict[key]['LASSO']['AngleVelocity']['weights']>0)[0]
        resultDict[key]['LASSO']['ConversePrediction'] = dr.runLinearModel(dataSets[key], resultDict[key], pars, splits, plot = True, behaviors = ['AngleVelocity', 'Eigenworm3'], fitmethod = 'LASSO', subset = subset)
#        # find non-linearity
        #dr.fitNonlinearity(dataSets[key], resultDict[key], splits, pars, fitmethod = 'LASSO', behaviors = ['AngleVelocity', 'Eigenworm3'])
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='LASSO', behaviors = behaviors, random = pars['trainingType'])
    plt.show()
    # predict opposites
    
    plt.show()
    

#%%
###############################################    
# 
# linear regression using elastic Net
#
##############################################
if elasticnet:
    for kindex, key in enumerate(keyList):
        print 'Running Elastic Net',  key
        splits = resultDict[key]['Training']
        resultDict[key]['ElasticNet'] = dr.runElasticNet(dataSets[key], pars,splits, plot=0, behaviors = behaviors)
        # calculate how much more neurons contribute
        tmpDict = dr.scoreModelProgression(dataSets[key], resultDict[key], splits,pars, fitmethod = 'ElasticNet', behaviors = behaviors, )
        for tmpKey in tmpDict.keys():
            resultDict[key]['ElasticNet'][tmpKey].update(tmpDict[tmpKey])
            
        # do converse calculation -- give it only the neurons non-zero in previous case
        subset = {}
        subset['AngleVelocity'] = np.where(np.abs(resultDict[key]['ElasticNet']['Eigenworm3']['weights'])>0)[0]
        subset['Eigenworm3'] = np.where(np.abs(resultDict[key]['ElasticNet']['AngleVelocity']['weights'])>0)[0]
        resultDict[key]['ElasticNet']['ConversePrediction'] = dr.runLinearModel(dataSets[key], resultDict[key], pars, splits, plot = True, behaviors = ['AngleVelocity', 'Eigenworm3'], fitmethod = 'ElasticNet', subset = subset)
        
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='ElasticNet', behaviors = behaviors,random = pars['trainingType'])
    plt.show()

 #%%
###############################################    
# 
# lag-time fits of neural activity
#
##############################################
if lagregression:
    for kindex, key in enumerate(keyList):
        print 'Running lag calculation',  key
        splits = resultDict[key]['Training']
        #resultDict[key]['LagLASSO'] = dr.timelagRegression(dataSets[key], pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], lags = np.arange(-18,19, 3))
        resultDict[key]['LagEN'] = dr.timelagRegression(dataSets[key], pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], lags = np.arange(-18,19, 3), flag='ElasticNet')
#%%
###############################################    
# 
#day-forward crossvalidation for test error estimates
#
##############################################
if nestedvalidation:
    for kindex, key in enumerate(keyList):
        print 'Running nested validation',  key
        splits = resultDict[key]['Training']
        resultDict[key]['nestedLASSO'] = dr.NestedRegression(dataSets[key], pars, splits, plot = False, behaviors = ['AngleVelocity', 'Eigenworm3'], flag = 'LASSO')    


#%%
###############################################    
# 
# overlay neuron projections with relevant neurons
#
##############################################
if positionweights:
    for kindex, key in enumerate(keyList):
        print 'plotting linear model weights on positions',  key
        
    mp.plotWeightLocations(dataSets, resultDict, keyList, fitmethod='ElasticNet')
    plt.show()
#%%
###############################################    
# 
# plot the number of neurons and scatter plot of predictions fo velocity and turns
#
##############################################
if resultsPredictionOverview:
    fitmethod = 'ElasticNet'
    mp.plotLinearModelScatter(dataSets, resultDict, keyList, pars, fitmethod=fitmethod, behaviors = ['AngleVelocity', 'Eigenworm3'], random = 'none')
    # collect the relevant number of neurons
    
    
    noNeur = []
    for key in keyList:
        noNeur.append([resultDict[key][fitmethod]['AngleVelocity']['noNeurons'], resultDict[key][fitmethod]['Eigenworm3']['noNeurons']])
    noNeur = np.array(noNeur)
    plt.figure()
    plt.bar([1,2], np.mean(noNeur, axis=0),yerr=np.std(noNeur, axis=0) )
    plt.scatter(np.ones(len(noNeur[:,0]))+0.5, noNeur[:,0])
    plt.scatter(np.ones(len(noNeur[:,0]))+1.5, noNeur[:,1])
    plt.xticks([1,2], ['velocity', 'Turns'])
    plt.show()
