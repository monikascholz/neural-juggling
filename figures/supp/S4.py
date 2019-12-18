# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 08:24:57 2018
fig1-S3 - estimate PCA noise level and covariances from reconstruction.
@author: monika
"""
import numpy as np
import matplotlib as mpl
import os
#
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.ticker as mtick
from scipy.stats import ttest_ind
from prediction import dataHandler as dh
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
# create figure 1: This is twice the normal size
#
################################################
fig = plt.figure('S4_PCA noise floor and cov reconstructions.', figsize=(4.25, 8.5))
gs1 = gridspec.GridSpec(4,1)#,  width_ratios=[1, 1,1,1])
gs1.update(left=0.15, right=0.95,  bottom = 0.07, top=0.95, hspace=0.35, wspace=0.25)
fig.patch.set_alpha(0.0)
#eigenvalue axes
ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[1,0])
ax3 = plt.subplot(gs1[2,0])
ax4 = plt.subplot(gs1[3,0])
##covariance axes
#ax11 = plt.subplot(gs1[1,0])
#ax21 = plt.subplot(gs1[1,1])
#ax31 = plt.subplot(gs1[1,2])
#ax41 = plt.subplot(gs1[1,3])
# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C', 'D']
locations = [(0, 0.95),  (0, 0.72),(0, 0.48), (0, 0.25)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
#letters = ['C', 'D']
#y0 = 0.45
#locations = [(0, y0), (0.5,y0), (0.72, y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
#            horizontalalignment='left',verticalalignment='baseline',)
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18', 'AML70', 'AML175']:
    for condition in [ 'chip', 'moving', 'immobilized']:# ['moving', 'immobilized', 'chip']:
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


#################################################
##
## Show eigenvalues from PCA
##
#################################################
# variance explained for moving and immobile 
nComp =10
movExp = ['AML32_moving', 'AML70_chip']
imExp = ['AML32_immobilized', 'AML70_immobilized']
movCtrl = ['AML18_moving', 'AML175_moving']
imCtrl = ['AML18_immobilized']

for condition, keys, ax in zip([ 'immobilized (GCaMP)','moving (GCaMP)', 'immobilized (GFP)','moving (GFP)'], [ imExp,movExp,imCtrl, movCtrl ],  [ax2, ax1, ax4, ax3]):
    tmpdata = []
    noiseS = []
    noiseL = []    
    for key in keys:
        dset = data[key]['analysis']
        for idn in dset.keys():
            results=  dset[idn]['PCA']
            
            tmpdata.append(results['eigenvalue'][:nComp])
            noiseS.append(results['fullShuffle'][:nComp])
            noiseL.append(results['lagShuffle'][:nComp])
          
        #ax1.plot(np.arange(1,nComp+1),np.array(tmpdata).T ,'-',color =colorsExp[condition], lw=1, label = '{} {}'.format(typ, condition),alpha=0.3 )
    
    ax.errorbar(np.arange(1,nComp+1), np.mean(noiseS, axis=0), np.std(noiseS, axis=0), color = 'k', marker='x', label= 'Shuffled')
    ax.errorbar(np.arange(1,nComp+1), np.mean(noiseL, axis=0), np.std(noiseL, axis=0), color = 'b', marker='o', label='Time-lag shuffle')
    ax.set_ylabel('Eigenvalues')
    ax.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = 'r', marker='s', label='Original PCA')    
    ax.set_title(condition)
    x0 = np.arange(1,nComp+1)[np.where((np.mean(tmpdata, axis=0)-np.mean(noiseL, axis=0))<0)[0][0]]
    
    t, p = ttest_ind(tmpdata, noiseL, axis=0, equal_var=False)
    x0 = np.where(p>0.05)[0][0]
    print condition, x0, len(noiseL)
    ax.axvline(x0, color='k', linestyle='--', zorder=-10)
    ax.set_yticks([0,25,50])
    ax.set_xticks([])
    #ax12.set_yticks([0,25,50,75,100])
ax4.set_xlabel('# of components')
ax4.set_xticks([1,5,10])
ax1.legend(loc=1, bbox_to_anchor=(0.55, 0.5, 0.5, 0.5))
plt.show()


##################################################
###
### Show covariance matrices
###
##################################################
## variance explained for moving and immobile 
#nComp =10
#movExp = ['AML32_moving', 'AML70_chip']
#imExp = ['AML32_immobilized', 'AML70_immobilized']
#movCtrl = ['AML18_moving', 'AML175_moving']
#imCtrl = ['AML18_immobilized']
#
#for condition, keys, ax in zip([ 'immobilized','moving', 'immobilized (Ctrl)','moving (Ctrl)'], [ imExp,movExp,imCtrl, movCtrl ],  [ax1, ax2, ax3, ax4]):
#    for key in keys:
#        dset = data[key]['analysis']
#        tmpdata = []
#        noiseS = []
#        noiseL = []
#        for idn in dset.keys():
#            results=  dset[idn]['PCA']
#            tmpdata.append(results['eigenvalue'][:nComp])
#            noiseS.append(results['fullShuffle'][:nComp])
#            noiseL.append(results['lagShuffle'][:nComp])
#          
#    ax.errorbar(np.arange(1,nComp+1), np.mean(noiseS, axis=0), np.std(noiseS, axis=0), color = 'k', marker='x')
#    ax.errorbar(np.arange(1,nComp+1), np.mean(noiseL, axis=0), np.std(noiseL, axis=0), color = 'b', marker='o')
#    ax.set_ylabel('Eigenvalues')
#    ax.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = 'r', marker='s')    
#    ax.set_title(condition)
#    x0 = np.arange(1,nComp+1)[np.where((np.mean(tmpdata, axis=0)-np.mean(noiseL, axis=0))>0)[-1][0]]
#    
#    #ax.set_yticks([0,25,50,75,100])
#    #ax12.set_yticks([0,25,50,75,100])
#    ax.set_xlabel('# of components')
#plt.show()