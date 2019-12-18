
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 2 - Behavior is represented in the brain
@author: monika
"""
import numpy as np
import matplotlib as mpl
import os
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter1d
#
#import makePlots as mp
from prediction import dataHandler as dh
from prediction import userTracker
# deliberate import all!
from prediction.stylesheet import *
from scipy.stats import pearsonr


# suddenly this isn't imported from stylesheet anymore...
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["font.size"] = 14
fs = mpl.rcParams["font.size"]
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18', 'AML175', 'AML70']:
    for condition in ['moving', 'chip']:# ['moving', 'immobilized', 'chip']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '{}_{}/'.format(typ, condition))
        dataLog = os.path.join(path,'{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
        outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
        outLocData = os.path.join(path,'/Analysis/{}_{}.hdf5'.format(typ, condition))
        
        try:
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
        except IOError:
            print typ, condition , 'not found.'
            pass
print 'Done reading data.'


################################################
#
# variance explained for our simple dataset
#
################################################
# select a special dataset - moving AML32
movingAML32 = 'BrainScanner20170613_134800'#'BrainScanner20170424_105620'#'
moving = data['AML32_moving']['input'][movingAML32]
movingAnalysis = data['AML32_moving']['analysis'][movingAML32]
label = 'AngleVelocity'
splits = movingAnalysis['Training']
train, test = splits[label]['Train'], splits[label]['Test']
# pull out repeated stuff
time = moving['Neurons']['TimeFull']
timeActual = moving['Neurons']['Time']
t = moving['Neurons']['Time'][test]
noNeurons = moving['Neurons']['Activity'].shape[0]
results = movingAnalysis['PCA']

X = moving['Neurons']['Activity']
print 'total variance', np.sum(np.var(X, axis=1))
wv = movingAnalysis['ElasticNet']['AngleVelocity']['weights']
wv = wv/np.linalg.norm(wv)
print np.var(np.dot(X.T, wv))
wbc = movingAnalysis['ElasticNet']['Eigenworm3']['weights']
wbc = wbc/np.linalg.norm(wbc)
print np.var(np.dot(X.T, wbc))
wpc1 = movingAnalysis['PCA']['neuronWeights'][:,0]
wpc1 = wpc1/np.linalg.norm(wpc1)
wpc2 = movingAnalysis['PCA']['neuronWeights'][:,1]
wpc2 = wpc2/np.linalg.norm(wpc2)
pc1, pc2 = movingAnalysis['PCA']['pcaComponents'][:2]
#plt.plot(np.dot(X.T, wv))
#plt.plot(np.dot(X.T, wbc))
comp = np.stack([wv, wbc])
v, bc = np.dot(X.T, wv), np.dot(X.T, wbc)
# resonstruct from PCA and SLM
#Xbeh = np.dot(comp.T,np.stack([v, bc]))
Xbeh = np.dot(comp.T,np.stack([v, bc]))

XPC = np.dot(np.stack([wpc1, wpc2]).T,np.stack([pc1, pc2]))
orderpc = movingAnalysis['PCA']['neuronOrderPCA']
order = np.argsort(wbc) 
plt.figure()
plt.subplot(221)
plt.scatter(np.dot(X.T, wv), np.dot(X.T, wbc), alpha=0.1, s=3)
plt.subplot(222)
plt.scatter(np.dot(X.T, wpc1), np.dot(X.T, wpc2), alpha=0.1, s=3)
plt.subplot(223)
plt.imshow(Xbeh[order], aspect = 'auto')
plt.subplot(224)
plt.imshow(XPC[orderpc], aspect='auto')
plt.show()
print X.shape
# variance explained for all 7 datasets
flag = 'ElasticNet'
gcamp = []

    
for key in ['AML32_moving', 'AML70_chip']:
    dset = data[key]['analysis']
    inputs = data[key]['input']
    
    for idn in dset.keys():
        varExp = []
        for behavior in ['AngleVelocity', 'Eigenworm3']:
            # z-scored activity
            X = inputs[idn]['Neurons']['Activity']
            # weight vector from SLM
            w = dset[idn][flag][behavior]['weights']
            # uncomment for testing - this should give the same var explained as PCA
            #w = dset[idn]['PCA']['neuronWeights'][:,0]
            # compare to variance explained pca
            pcaVar = dset[idn]['PCA']['expVariance'][:2]
            print pcaVar
            # normalize vector
            w = w/np.linalg.norm(w)
            # store variance, total variance
            varExp.append([np.var(np.dot(X.T, w)),  np.sum(np.var(X, axis=1)), pcaVar[0]])
        gcamp.append(varExp)

gcamp = np.array(gcamp)
print gcamp
print gcamp[:,0,0]/gcamp[:,0, 1]
print gcamp[:,1,0]/gcamp[:,1, 1]

plt.figure(figsize=(3,4))
ax = plt.subplot(111)
labels = [r'PC$_1$', 'velocity', 'body curvature']
mkStyledBoxplot(ax, [0, 1, 2], [gcamp[:,0,2],gcamp[:,0,0]/gcamp[:,0, 1],  gcamp[:,1,0]/gcamp[:,1, 1]], [N0, R1, B1], labels)
ax.set_ylabel('Variance explained')
ax.set_xlim([-0.5, 2.25])
plt.tight_layout()
plt.show()