#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:56:27 2018
show R2 plots for training sets to show we are not dominated by extremes. And show PCA versus EM weights.
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
import prediction.dataHandler as dh
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
for typ in ['AML32', 'AML70']:
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

fig = plt.figure('S6_PCA and ENet weights', figsize=(9.5,7.5))
# this is a gridspec
gs1 = gridspec.GridSpec(4, 3, width_ratios=[2,1,2])
gs1.update(left=0.09, right=0.97,  bottom = 0.07, top=0.95, hspace=0.45, wspace=0.25)

# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C']
y0 = 0.96
locations = [(0,y0),  (0.4,y0), (0.65,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)



######
axBehPCA =plt.subplot(gs1[0, 0])
axBehEN =plt.subplot(gs1[1, 0])

axBehPCAbc =plt.subplot(gs1[2, 0])
axBehENbc =plt.subplot(gs1[3, 0])
######
axPCAfit = plt.subplot(gs1[0, 1])
axENfit = plt.subplot(gs1[1, 1])

axPCAfitbc = plt.subplot(gs1[2, 1])
axENfitbc = plt.subplot(gs1[3, 1])
# PCA and LASSO weight histograms
gsWeights = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs1[:,2:],hspace=0.5, wspace=0.35)

axes = []
label = 'AngleVelocity'
movingAML32 = 'BrainScanner20170613_134800'

moving = data['AML32_moving']['input'][movingAML32]
movingAnalysis = data['AML32_moving']['analysis'][movingAML32]
splits = movingAnalysis['Training']
train, test = splits[label]['Train'], splits[label]['Test']
#XXXX velocity correction
velocity = moving['Behavior'][label]
bc = moving['Behavior']['Eigenworm3']
# pull out repeated stuff
time = moving['Neurons']['TimeFull']
timeActual = moving['Neurons']['Time']
t = moving['Neurons']['Time'][test]
noNeurons = moving['Neurons']['Activity'].shape[0]
resultsPCA = movingAnalysis['PCA']

ypredEN = movingAnalysis['ElasticNet'][label]['output']
ypredPCA = movingAnalysis['PCAPred'][label]['output']
# rescale ypred because we didn't do so in the prediction code
ypredPCA = (ypredPCA *np.std(velocity))+ np.mean(velocity)

# body curvature prediction
ypredENbc = movingAnalysis['ElasticNet']['Eigenworm3']['output']
ypredPCAbc = movingAnalysis['PCAPred']['Eigenworm3']['output']
ypredPCAbc = ypredPCAbc*np.std(bc) + np.mean(bc)

# PCA weights
weightsPCA = resultsPCA['neuronWeights']
print weightsPCA.shape
# PCAorder
PCAorder = movingAnalysis['PCA']['neuronOrderPCA']

# EN weights
weightsENv = movingAnalysis['ElasticNet'][label]['weights']
print weightsENv
print movingAnalysis['ElasticNet'][label]['noNeurons']
weightsENbc = movingAnalysis['ElasticNet']['Eigenworm3']['weights']
####################
# plot velocity prediction
#########################
for ax, pred in zip([axBehPCA,axBehEN], [ypredPCA, ypredEN]):
    ax.plot(timeActual, velocity, color=N1)
    ax.plot(timeActual, pred, color=R1)
    # draw a box for the testset
    ax.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
    ax.set_xticks([])


for ax, pred in zip([axPCAfit, axENfit], [ypredPCA, ypredEN]):
    ax.scatter( velocity[test], pred[test], color=R1, alpha=0.05, s = 10)
    # draw a line for the linear fit
    ax.plot([np.min(velocity),np.max(velocity)], [np.min(velocity),np.max(velocity)], 'k--')

axBehPCA.set_title('PCA')
axBehEN.set_title('SLM ')

axBehPCAbc.set_title('PCA')
axBehENbc.set_title('SLM')

####################
# plot turn prediction
#########################
for ax, pred in zip([axBehPCAbc,axBehENbc], [ypredPCAbc, ypredENbc]):
    ax.plot(timeActual, bc, color=N1)
    ax.plot(timeActual, pred, color=B1)
    # draw a box for the testset
    ax.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)

for ax, pred in zip([axPCAfitbc, axENfitbc], [ypredPCAbc, ypredENbc]):
    ax.scatter(bc[test], pred[test], color=B1, alpha=0.05, s= 10)
    # draw a line for the linear fit
    ax.plot([np.min(bc),np.max(bc)], [np.min(bc),np.max(bc)], 'k--')


axBehPCAbc.set_xticks([])
axENfitbc.set_xlabel('True')
fig.text(0.4, 0.5, "Predicted", rotation = 90, verticalalignment='center')
axBehENbc.set_xlabel('Time (s)')
fig.text(0.01, 0.75, "velocity (rad/s)", rotation = 90, verticalalignment='center', fontsize=fs)
fig.text(0.01, 0.25, "Body curvature", rotation = 90, verticalalignment='center')
    
####################
# plot PCA and elastic net weights
#########################
axes = []
bins = np.arange(len(PCAorder))
for i in range(3):
    # left column
    ax = plt.subplot(gsWeights[i, 0])
    ax.fill_between(bins, weightsPCA[PCAorder, i], step = 'pre', color=N1)
    axes.append(ax)
    # right column
#    ax2 = plt.subplot(gsWeights[i, 1], sharey = ax)
#    ax2.fill_between(bins, weightsPCA[PCAorder, i], step = 'pre', color=B1)
#    
#    plt.setp(ax2.get_yticklabels(), visible=False)
#    axes.append(ax2)
    
# Elastic Net weights
ax = plt.subplot(gsWeights[3, 0])
ax.fill_between(bins, weightsENv[PCAorder], step = 'pre', color=R1, alpha=0.5)
axes.append(ax)
ax.fill_between(bins, weightsENbc[PCAorder], step = 'pre', color=B1, alpha=0.5)

#axes[-2].set_xlabel('Neurons')
axes[-1].set_xlabel('Neurons')

titles = [r'PC$_1$', r'PC$_2$', r'PC$_3$', 'SLM']

for ax, title in zip(axes, titles):
    ax.set_title(title)
for ax in axes[:-2]:
    plt.setp(ax.get_xticklabels(),visible=False)
#axes[1].set_title('Body \n curvature')
fig.text(0.65, 0.5, "Weights", rotation = 90, verticalalignment='center')
for ax in axes:
    moveAxes(ax, 'scalex', -0.04)
    moveAxes(ax, 'right', 0.04)
plt.show()
