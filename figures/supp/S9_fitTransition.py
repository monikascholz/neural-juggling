
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 2 - Behavior is represented in the brain
@author: monika
"""
import numpy as np
#import matplotlib as mpl
import os
#
import matplotlib.style
import matplotlib as mpl

# deliberate import all!
from prediction.stylesheet import *
import matplotlib.gridspec as gridspec
from scipy.ndimage.filters import gaussian_filter1d
import prediction.dataHandler as dh
from prediction import userTracker
from sklearn import linear_model
from sklearn import preprocessing

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
for typ in ['Special']:
    for condition in ['transition']:# ['moving', 'immobilized', 'chip']:
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
# gte some data out
#
################################################
# select a special dataset - transiently immobilized
transientData = 'BrainScanner20180511_134913'
transient = data['Special_transition']['input']['BrainScanner20180511_134913']
transientAnalysis = data['Special_transition']['analysis']['BrainScanner20180511_134913']

# PCA of immobile part
pcaImm = transientAnalysis['PCAHalf2']
pcs =  pcaImm['fullData']
# behaviors to predict
av = transient['Behavior']['AngleVelocity']
t = transient['Behavior']['Eigenworm3']

# get a time axis.
time = transient ['Neurons']['TimeFull']
timeActual = transient ['Neurons']['Time']
# create testset - 3 minutes to fit, one to predict
# get original testset.
a,b = 2,3
origTest = transientAnalysis['Training']['Half']['Train']
origTrain = transientAnalysis['Training']['Half']['Test']
train = origTest[:int(a*max(origTest)/b)]
test = origTest[int(a*max(origTest)/b):max(origTest)]
ncomp = 5
################################################
#
# run fit on moving part using immobile PCA axes
#
################################################
pcamodel = []
r2s = []
#for ncomp in range(3,20):
for behavior in [av, t]:
    # scale behavior to same 
    #behavior = preprocessing.scale(behavior)
    #now we use a linear model to train and test our predictions
    # lets build a linear model
    lin = linear_model.LinearRegression(normalize=False)
    X = pcs[:ncomp].T
    lin.fit(X[train], behavior[train])

    score = lin.score(X[train],behavior[train])
    scorepred = lin.score(X[test], behavior[test])
    #print 'PCA prediction results:'
    print score,  scorepred, ncomp,
    pcamodel.append(lin.predict(X))
    r2s.append(scorepred)
print '\n'

################################################
#
# set up figure
#
################################################
fig = plt.figure('S9_Prediction on transition set', figsize=(4.5, 6.5))
gs1 = gridspec.GridSpec(5,1,  width_ratios=[1], height_ratios = [0.15,0.1, 2,1,1])
gs1.update(left=0.2, right=0.98,  bottom = 0.08, top=0.98, hspace=0.2, wspace=0.75)

ax1 = plt.subplot(gs1[0,0])
axTetra = plt.subplot(gs1[1,0])
ax2 = plt.subplot(gs1[2,0])
ax3 = plt.subplot(gs1[3,0])
ax4 = plt.subplot(gs1[4,0])

letters = ['A', 'B', 'C', 'D']
x0 = 0
locations = [(x0,0.95),  (x0,0.9), (x0,0.54),  (x0,0.25)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
################################################
#
# plot results
#
################################################
# add tetramisole
yloc = 1
axTetra.text(np.mean(timeActual[train[-1]]), 0.94*yloc, "+ paralytic",horizontalalignment='left', color='k', fontsize=fs)
# the most complicated way to get a step drawn
axTetra.step([timeActual[origTest[-1]],timeActual[origTrain[-1]]], [0.92*yloc, 0.92*yloc], color='k', linestyle='-')
axTetra.plot([timeActual[origTest[0]],timeActual[origTest[-1]]], [0.86*yloc, 0.86*yloc], color='k', linestyle='-')
axTetra.plot([timeActual[origTest[-1]],timeActual[origTest[-1]]], [0.86*yloc, 0.92*yloc], color='k', linestyle='-')         
cleanAxes(axTetra)
axTetra.set_xlim([np.min(timeActual), np.max(timeActual)])
moveAxes(axTetra, 'down', 0.03)
# plot test, train and immobilized set
y0 = 0.5#ax2.get_ylim()[-1]/2.

ax1.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N1, zorder=-10, alpha=0.75)
ax1.text(np.mean(timeActual[test]), y0, 'Test',horizontalalignment='center', verticalalignment='center')
ax1.axvspan(timeActual[train[0]], timeActual[train[-1]], color=N2, zorder=-10, alpha=0.75)
ax1.text(np.mean(timeActual[train]), y0, 'Train',horizontalalignment='center', verticalalignment='center')
ax1.axvspan(timeActual[origTrain[0]], timeActual[origTrain[-1]], color='orange', zorder=-10, alpha=0.75)
ax1.text((timeActual[origTrain[0]]+timeActual[origTrain[-1]])/2.,y0 , 'find PCs',horizontalalignment='center', verticalalignment='center')


# plot PCA components
for i in range(ncomp):
    #y = results['pcaComponents'][i]
    y = pcs[i]
    # normalize
    y =y -np.min(y)
    y =y/np.max(y)
    ax2.text(-100, np.mean(y)+i*1.05, r'PC$_{{{}}}$'.format(i+1), color = 'k')
    ax2.plot(time[transient['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = 'k')

for ax in [ax1, ax2]:
    ax.set_xlim([np.min(timeActual), np.max(timeActual)])
    cleanAxes(ax)
    
    
# plot behaviors
ax3.plot(timeActual, av, N0, lw=2, label = 'Measured velocity')    
ax4.plot(timeActual, t, N0, lw=2, label = 'Measured body curvature')

# plot Predictions
ax3.plot(timeActual, pcamodel[0], R1,  label = 'Predicted velocity') 
ax4.plot(timeActual, pcamodel[1], B1, label = 'Predicted body curvature')

# repeat testset
ax3.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N1, zorder=-10, alpha=0.75)
ax4.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N1, zorder=-10, alpha=0.75)

# write R2 values
ax3.text(timeActual[test[0]], ax3.get_ylim()[-1], "R$^2=${0:.2f}".format(r2s[0]))
ax4.text(timeActual[test[0]],  ax4.get_ylim()[-1], "R$^2=${0:.2f}".format(r2s[1]))

#label axes
ax3.set_ylabel('Velocity (rad/s)')
ax4.set_ylabel('Body \n curvature')
ax4.set_xlabel('Time (s)')

for ax in [ax3, ax4]:
    ax.set_xlim([np.min(timeActual), np.max(timeActual)])
ax3.set_xticks([])

plt.show()

