
# standard modules
import numpy as np
import matplotlib.pylab as plt
import h5py
import os
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr
from prediction import userTracker
    

###############################################    
# 
#   datset params
#
###############################################
typ = 'AML70' # possible values AML32, AML18, AML70
condition = 'moving' # Moving, immobilized, chip
overview = True
###############################################    
# 
#    load data into dictionary
#
##############################################
data = {}
for typ in ['AML32', 'AML70']:
    for condition in ['moving', 'immobilized']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '{}_{}/'.format(typ, condition))
        dataLog = os.path.join(path,'{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
        outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
        outLocData = os.path.join(path,'/Analysis/{}_{}.hdf5'.format(typ, condition))
        
        
        # load multiple datasets
        dataSets = dh.loadDictFromHDF(outLocData)
        keyList = np.sort(dataSets.keys())
        results = dh.loadDictFromHDF(outLoc) 
        # store in dictionary by typ and condition
        key = '{}_{}'.format(typ, condition)
        data[key] = {}
        data[key]['dsets'] = keyList
        data[key]['input'] = dataSets
        data[key]['analysis'] = results
        

# analysis parameters
behaviors = ['AngleVelocity', 'Eigenworm3']

colors = {'moving': '#204a87ff',
            'immobilized': '#cc0000ff'}

###############################################    
# 
# plot PCA dimensionality of moving versus immobile
#
##############################################
fig = plt.figure('Compare Dimensionality')
ax4 = plt.subplot(111)
for typ in ['AML70']:
    for condition in ['moving', 'immobilized']:
        key = '{}_{}'.format(typ, condition)
        dset = data[key]['analysis']
        #
        tmpdata = []
        for idn in dset.keys():
            results=  dset[idn]['PCA'] 
            nComp = dset[idn]['PCA'] ['nComp']
            rescale = 1.0*data[key]['input'][idn]['Neurons']['Activity'].shape[0]
            #rescale = 1
            ax4.plot(np.arange(1,nComp+1)/rescale,np.cumsum(results['expVariance'])*100, '-',alpha = 0.75,color = colors[condition], lw=1)
            tmpdata.append(np.cumsum(results['expVariance'])*100)
        ax4.plot(np.arange(1,nComp+1),np.cumsum(results['expVariance'])*100 ,'o-',color = colors[condition], lw=1, label = '{} {}'.format(typ, condition))
        ax4.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = colors[condition])
    ax4.set_ylabel('Explained variance (%)')
    ax4.set_yticks([0,25,50,75,100])
    ax4.set_xlabel('Number of components')
    plt.legend()
    fig.add_subplot(ax4)
plt



###############################################    
# 
# some generic data checking plots
#
##############################################
if overview:
    #mp.plotBehaviorAverages(dataSets, keyList) 
    mp.plotVelocityTurns(dataSets, keyList)
    mp.plotDataOverview(dataSets, keyList)
    mp.plotNeurons3D(dataSets, keyList, threed = False)  
    #mp.plotExampleCenterlines(dataSets, keyList, folder)
    plt.show() 
    
###############################################    
# 
# average results
#
##############################################
mp.averageResultsLinear(resultDict,resultDictCtrl, keyList,keyListCtrl, fitmethod = "LASSO",  behaviors = ['AngleVelocity', 'Eigenworm3'])
plt.show()
#mp.averageResultsLinear(resultDict, resultDictCtrl, keyList,keyListCtrl, fitmethod = "ElasticNet",  behaviors = ['AngleVelocity', 'Eigenworm3'])
#plt.show()
labelsPCA = ['GCamp6s', 'GFP', 'GCamp6s immobilized']
colorsPCA = ['#007398', 'k', '#24c8ff']
res, keys = [resultDict, resultDictCtrl,resultDictimm],[keyList,keyListCtrl, keyListimm]
mp.averageResultsPCA(res, keys, labelsPCA,colorsPCA,fitmethod = "PCA")
plt.show()
###############################################    
# 
# use svm to predict discrete behaviors
#
##############################################
if svm:
    
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
    
    # overview of PCA results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList, pars)
    plt.show()
    # show correlates of PCA
    mp.plotPCAcorrelates(dataSets, resultDict, keyList, pars, flag='PCA')
    #  plot 3D trajectory of PCA
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'time')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'velocity')
    plt.show()
    mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'turns')
    plt.show()
    #%%
    ###############################################    
    # 
    # overlay neuron projections with relevant neurons
    #
    ##############################################
    # TODO change to make weights larger or stth.
    
        
    #mp.plotWeightLocations(dataSets, resultDict, keyList, fitmethod='PCA')
    #plt.show()
    #%%
  
#%%
###############################################    
# 
# linear regression using LASSO
#
##############################################
if lasso:
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='LASSO', behaviors = behaviors, random = pars['trainingType'])
    plt.show()
    mp.plotLinearModelResults(dataSetsCtrl, resultDictCtrl, keyListCtrl[:2], pars, fitmethod='LASSO', behaviors = behaviors, random = pars['trainingType'])
    plt.show()
    # overview of LASSO results and weights
    mp.plotPCAresults(dataSets, resultDict, keyList, pars,  flag = 'LASSO')
    plt.show()
    #  plot 3D trajectory of SVM
    #mp.plotPCAresults3D(dataSets, resultDict, keyList, pars, col = 'etho', flag = 'LASSO')
    plt.show()
    
    
    
#%%
###############################################    
# 
# linear regression using elastic Net
#
##############################################
if elasticnet:
    
    mp.plotLinearModelResults(dataSets, resultDict, keyList, pars, fitmethod='ElasticNet', behaviors = behaviors,random = pars['trainingType'])
    plt.show()


#%%
###############################################    
# 
# overlay neuron projections with relevant neurons
#
##############################################
if positionweights:
   
    mp.plotWeightLocations(dataSets, resultDict, keyList, fitmethod='LASSO')
    plt.show()
#%%
###############################################    
# 
# plot the number of neurons and scatter plot of predictions fo velocity and turns
#
##############################################
if resultsPredictionOverview:
    fitmethod = 'LASSO'
    mp.plotLinearModelScatter(dataSets, resultDict, keyList, pars, fitmethod='LASSO', behaviors = ['AngleVelocity', 'Eigenworm3'], random = 'none')
    # collect the relevant number of neurons
    
    
    noNeur = []
    for key in keyList:
        noNeur.append([resultDict[key]['LASSO']['AngleVelocity']['noNeurons'], resultDict[key]['LASSO']['Eigenworm3']['noNeurons']])
    noNeur = np.array(noNeur)
    plt.figure()
    plt.bar([1,2], np.mean(noNeur, axis=0),yerr=np.std(noNeur, axis=0) )

    