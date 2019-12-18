#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:56:27 2018
Show R2 for temporal shifts between neural actiity and behavior.
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

fig = plt.figure('S8_Lag_plots', figsize=(9.5,4.75))
# this is a gridspec
gs1 = gridspec.GridSpec(2, 4)
gs1.update(left=0.09, right=0.99,  bottom = 0.1, top=0.95, hspace=0.25, wspace=0.55)

## add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C', 'D']
y0 = 0.96
locations = [(0,y0),  (0.26,y0), (0.51,y0), (0.71,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
letters = ['E', 'F', 'G', 'H']
y0 = 0.46
locations = [(0,y0),  (0.26,y0), (0.51,y0), (0.71,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)


####################
# get lagged r2s
#########################
a = 0.4
behaviors = ['AngleVelocity', 'Eigenworm3']
colors = [R1, B1]
for i in range(2):
    r2s = []
    ns = []
    ax = plt.subplot(gs1[i, 0])
    ax1 = plt.subplot(gs1[i, 1])
    ax2 = plt.subplot(gs1[i, 2])
    ax3 = plt.subplot(gs1[i, 3])
    ax1.axvline(0, color='k', linestyle = '--')
    ax2.axvline(0, color='k', linestyle = '--')
    ax3.axvline(0, color='k', linestyle = '--')
    ax.axvline(0, color='k', linestyle = '--')
    ax.set_ylabel(r'R$^2$')
    ax1.set_ylabel(r'R$^2$/R$^2_{max}$')
    ax2.set_ylabel(r'# Neurons')
    ax3.set_ylabel(r'# Neurons (norm.)')
    
    ax.set_yticks([-0.2, 0., 0.2, 0.4, 0.6])
    ax3.set_yticks([0, 0.5, 1])
            
                       
    for key in ['AML32_moving', 'AML70_chip']:
        for dset in data[key]['analysis'].values():
            
            lags = dset['LagEN'][behaviors[i]]['lags']/6. #in seconds
            r2 = dset['LagEN'][behaviors[i]]['scorepredicted']
            n = dset['LagEN'][behaviors[i]]['noNeurons']
            r2s.append(r2)
            ns.append(n)
#            # plot raw r2s
#            ax.plot(lags, r2, colors[i], alpha=a)
#            # plot normalized R2
#            ax1.plot(lags, r2/np.max(r2), colors[i], alpha=a)
#            # plot raw ns
#            ax2.plot(lags, n, colors[i], alpha=a)
#            # plot normalized ns
#            ax3.plot(lags, n/1.0/np.max(n), colors[i], alpha=a)
    r2s = np.array(r2s)
    ns = np.array(ns)
    # plot the mean of things
    m = np.mean(r2s, axis = 0)
    sd = np.std(r2s, axis = 0)
    ax.plot(lags, m, colors[i], lw=2)
    ax.fill_between(lags, m-sd,m+sd, color = colors[i], alpha=0.2)
    
     
    # plot the mean of things
    m = np.mean(r2s/np.max(r2s, axis=1)[:,None], axis = 0)
    sd = np.std(r2s/np.max(r2s, axis=1)[:,None], axis = 0)
    ax1.plot(lags, m, colors[i], lw=2)
    ax1.fill_between(lags, m-sd,m+sd, color = colors[i], alpha=0.2)
    
     # plot the mean of things
    m = np.mean(ns, axis = 0)
    sd = np.std(ns, axis = 0)
    ax2.plot(lags, m, colors[i], lw=2)
    ax2.fill_between(lags, m-sd,m+sd, color = colors[i], alpha=0.2)
    
     # plot the mean of things
    m = np.mean(ns/1.0/np.max(ns, axis=1)[:,None], axis = 0)
    sd = np.std(ns/1.0/np.max(ns, axis=1)[:,None], axis = 0)
    ax3.plot(lags, m, colors[i], lw=2)
    ax3.fill_between(lags, m-sd,m+sd, color = colors[i], alpha=0.2)
    
    for axis in [ax, ax1, ax2, ax3]:
        if i ==1:
            axis.set_xlabel('Lag (s)')
        
        axis.set_xlim([-3,3])
        axis.set_xticks([-3,0,3])
plt.show()

