
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure 2 - Behavior is represented in the brain
@author: monika
"""
import numpy as np
import matplotlib as mpl
from sklearn.feature_selection import mutual_info_classif
import os
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import prediction.dataHandler as dh
from prediction import userTracker
# deliberate import all!
from prediction.stylesheet import *
# suddenly this isn't imported from stylesheet anymore...
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["font.size"] = 12
fs = mpl.rcParams["font.size"]
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML18', 'AML32']:
    for condition in ['immobilized']:# ['moving', 'immobilized', 'chip']:
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
# create figure 1: This is twice the normal size
#
################################################
# we will select a 'special' dataset here, which will have all the individual plots
fig = plt.figure('FigManifolds', figsize=(7.5, 3.5))
# this gridspec makes one example plot of a heatmap with its PCA
#gs1 = gridspec.GridSpec(4, 3, width_ratios = [1,1,1], height_ratios=[0.1, 1,1,2])
#gsHeatmap = gridspec.GridSpecFromSubplotSpec(4,5, subplot_spec=gs1[0:4,:], width_ratios=[1.25, 0.1, 0.5,0.5,0.5], height_ratios = [0.1,10,10,10], wspace=0.3, hspace=0.25)
gs1 = gridspec.GridSpec(1,2)
gs1.update(left=0.1, right=0.98,  bottom = 0.07, top=0.99, hspace=0.45, wspace=0.25)
################################################
#
# letters
#
################################################

# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B']
y0 = 0.99
locations = [(0,y0),  (0.5,y0), (0.62,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='top',)

################################################
#
#first row -- Signal analysis
#
################################################
# suddenly this isn't imported from stylesheet anymore...
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["font.size"] = 12
fs = mpl.rcParams["font.size"]

#################################################
##
## plot beautiful manifold!
##
#################################################

dset = data['AML32_immobilized']['analysis']
for di, key in enumerate(dset.keys()[:1]):
    print key
    # pull out all the components
    x, y, z = dset[key]['PCA']['fullData'][:3]
    
    # normalize components
    x/=np.max(x)
    y/=np.max(y)
    z/=np.max(z)
    # smooth
    # make smoooth
    smooth = 12
    from scipy.ndimage.filters import gaussian_filter1d
    x = gaussian_filter1d(x, smooth)
    y = gaussian_filter1d(y, smooth)
    z = gaussian_filter1d(z, smooth)
    # plot in 3d
    ax = plt.subplot(gs1[di], projection='3d', clip_on = False, zorder=-100, aspect='equal')
    ax.plot(x, y, z, color=N1, zorder=-10)
    ax.scatter(x[::12],y[::12],z[::12], color=B1, s=2)
    ax.set_xlabel(r'PC$_1$', labelpad = 5)
    ax.set_ylabel(r'PC$_2$', labelpad = 5)
    ax.set_zlabel(r'PC$_3$', labelpad = 5)
    #ax.view_init(elev=40, azim=150)
    ax.view_init(elev=-15, azim=-70)
    
    moveAxes(ax, 'scale', 0.025)
    moveAxes(ax, 'left', 0.05)
    moveAxes(ax, 'down', 0.1)
    
    ax = plt.subplot(gs1[di+1], projection='3d', clip_on = False, zorder=-100, aspect='equal')
    ax.plot(x, y, z, color=N1, zorder=-10)
    ax.scatter(x[::12],y[::12],z[::12], color=B1, s=2)
    moveAxes(ax, 'scale', 0.025)
    moveAxes(ax, 'left', 0.05)
    moveAxes(ax, 'down', -0.025)
   
    ax.set_xlabel(r'PC$_1$', labelpad = 5)
    ax.set_ylabel(r'PC$_2$', labelpad = 5)
    ax.set_zlabel(r'PC$_3$', labelpad = 5)


    #l = np.zeros(3)#+axmin

    
    #ax.view_init(elev=-40, azim=90)
plt.show()
