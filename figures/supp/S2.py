
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure Fig1-S2 - Heatmap examples for moving, immobilized and transition.
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
import prediction.dataHandler as dh
from prediction import userTracker
# deliberate import all!
from prediction.stylesheet import *
# suddenly this isn't imported from stylesheet anymore...
#mpl.rcParams["axes.labelsize"] = 14
#mpl.rcParams["xtick.labelsize"] = 14
#mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["font.size"] = 12
fs = mpl.rcParams["font.size"]
################################################
#
# grab all the data we will need
#
################################################

data = {}
for typ in ['AML32', 'AML18', 'AML70', 'AML175', 'Special']:
    for condition in ['chip', 'moving', 'immobilized', 'transition']:# ['moving', 'immobilized', 'chip']:
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
fig = plt.figure('S2-multipleExampleDatasets', figsize=(9.5, 9*2/2.), dpi=100)
gsHeatmap = gridspec.GridSpec(6,3,  width_ratios=[1, 1, 0.1], height_ratios = [1,0.1,0.75,1,0.1,0.75])
gsHeatmap.update(left=0.055, right=0.99,  bottom = 0.05, top=0.9, hspace=0.25, wspace=0.45)
#fig.patch.set_alpha(0.0)
#heatmap axes
axhm1 = plt.subplot(gsHeatmap[0,0])
axhm2 = plt.subplot(gsHeatmap[0,1])
#axhm3 = plt.subplot(gsHeatmap[0,2])
axcb = plt.subplot(gsHeatmap[0,2])
# ethogram
axetho1 = plt.subplot(gsHeatmap[1,0], clip_on=False)
axetho2 = plt.subplot(gsHeatmap[1,1], clip_on=False)
#axetho3 = plt.subplot(gsHeatmap[1,2], clip_on=False)
# legend for ethogram
axEthoLeg = plt.subplot(gsHeatmap[1,2])#,clip_on=False)
# principal components
axpc1 =plt.subplot(gsHeatmap[2,0], clip_on=False)#, sharex=axhm)
axpc2 =plt.subplot(gsHeatmap[2,1], clip_on=False)#, sharex=axhm)

#axpc3 =plt.subplot(gsHeatmap[2,2], clip_on=False)#, sharex=axhm)


############## for controls
#heatmap axes
axhm1a = plt.subplot(gsHeatmap[3,0])
axhm2a = plt.subplot(gsHeatmap[3,1])
#axhm3a = plt.subplot(gsHeatmap[3,2])
#axcba = plt.subplot(gsHeatmap[3,3])
# ethogram
axetho1a = plt.subplot(gsHeatmap[4,0], clip_on=False)
axetho2a = plt.subplot(gsHeatmap[4,1], clip_on=False)
#axetho3a = plt.subplot(gsHeatmap[4,2], clip_on=False)
# legend for ethogram
#axEthoLega = plt.subplot(gsHeatmap[4,3])#,clip_on=False)
# principal components
axpc1a =plt.subplot(gsHeatmap[5,0], clip_on=False)#, sharex=axhm)
axpc2a =plt.subplot(gsHeatmap[5,1], clip_on=False)#, sharex=axhm)
#axpc3a =plt.subplot(gsHeatmap[5,2], clip_on=False)#, sharex=axhm)
for ax in [axpc1, axpc2, axpc1a,axpc2a]:
    moveAxes(ax, 'up', 0.03)
for ax in [axpc1, axpc2, axetho1,axetho2, axhm1, axhm2]:
    moveAxes(ax, 'up', 0.04)
#for ax in [axhm1a, axhm2a]:
#    moveAxes(ax, 'down', 0.03)
#moveAxes(axpc2, 'up', 0.05)
#plt.show()
################################################
#
# letters
#
################################################
## mark locations on the figure to get good guess for a,b,c locs
#for y in np.arange(0,1.1,0.1):
#    plt.figtext(0, y, y)
#for x in np.arange(0,1.1,0.1):
#    plt.figtext(x, 0.95, x)
#
# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C', 'D', 'E', 'F']
x0 = 0
locations = [(x0,0.95),  (x0,0.70), (x0,0.66), (x0,0.48),  (x0,0.23), (x0,0.18)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['G', 'H', 'I', 'J', 'K', 'L']
x0 = 0.45
locations = [(x0,0.95),  (x0,0.70), (x0,0.66), (x0,0.48),  (x0,0.23), (x0,0.18)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)

################################################
#
#first row -- show example heatmaps for a movng and immobile dataset
#
################################################
movData = 'BrainScanner20170424_105620'
immData = 'BrainScanner20180510_092218'
#transientData = 'BrainScanner20180329_152141'
flag = 'PCA'
for key, axhm,axetho, axpc, dset, title  in zip([movData, immData], \
    [axhm1, axhm2], [axetho1, axetho2],[axpc1, axpc2],\
    [data['AML32_moving'],data['AML32_immobilized']],\
    ['moving', 'immobilized']):
    
    # get data
    transient = dset['input'][key]
    results2half = dset['analysis'][key][flag]
    time = transient['Neurons']['TimeFull']
    timeActual = transient['Neurons']['Time']
    #heatmap
#    data1 = transient['Neurons']['RawActivity'][results2half['neuronOrderPCA']]
#    data2 = transient['Neurons']['Ratio'][results2half['neuronOrderPCA']]
#    print np.nanmax(data1), np.nanmax(data2)
#    cax1 = plotHeatmap(timeActual, data1/np.nanmax(data1)-data2/np.nanmax(data2), ax=axhm, vmin=-1, vmax=1)
    data1 = transient['Neurons']['ActivityFull'][results2half['neuronOrderPCA']]
    cax1 = plotHeatmap(time, data1, ax=axhm, vmin=-2, vmax=2)
    axhm.xaxis.label.set_visible(False)
    axhm.set_xticks([])
    axhm.set_title(title)
    #ethogram
    plotEthogram(axetho, time, transient['Behavior']['EthogramFull'], alpha = 1, yValMax=1, yValMin=0, legend=0)
    cleanAxes(axetho, 'all')
    moveAxes(axetho, 'scaley', 0.02)
    axetho.xaxis.label.set_visible(False)
    
    # plot PCA components
    for i in range(np.min([len(results2half['pcaComponents']), 3])):
        #y = results['pcaComponents'][i]
        y = results2half['fullData'][i]
        # normalize
        y =y -np.min(y)
        y =y/np.max(y)
        if key == 'BrainScanner20170613_134800':
            axpc.text(-100, np.mean(y)+i*1.05, 'PC{}'.format(i+1), color = 'k')
        axpc.plot(time[transient['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = 'k')
    
    yloc = axpc.get_ylim()[-1]*1.1
    # labels and such
    #axpc.set_xlabel('Time (s)')
    axpc.set_xlim([np.min(timeActual), np.max(timeActual)])
    axpc.set_ylim([axpc.get_ylim()[0], yloc*1.01])
    cleanAxes(axpc, where='y')
    moveAxes(axpc, 'down', 0.02)
## legend for ethogram
#moveAxes(axEthoLeg, 'right', 0.025)
moveAxes(axEthoLeg, 'up', 0.02)
cleanAxes(axEthoLeg, where='all')

handles, labels = axetho.get_legend_handles_labels()
leg = mpl.legend.Legend(axEthoLeg, handles[::-1], labels[::-1],frameon=1, loc=1,prop={'size':12},handlelength=0.5, labelspacing=0,handletextpad=0.5)#,bbox_to_anchor=(-1, 0.9), loc=9)
for hndl in leg.legendHandles:
    hndl._sizes = [0]
axEthoLeg.add_artist(leg);

 # colorbar
cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
cbar.set_ticks([-2,2])
cbar.set_ticklabels(['<-2','>2'])
cbar.outline.set_visible(False)
moveAxes(axcb, 'left', 0.06)
moveAxes(axcb, 'scaley', -0.08)
axcb.set_ylabel(r'$\Delta I/I_0$', labelpad = -5, rotation=-90)
################################################
#
#second row -- show example heatmaps for a movng and immobile dataset
# this is for GFP
################################################
movData = 'BrainScanner20160506_160928'
immData = 'BrainScanner20180518_094052'
    
for key, axhm,axetho, axpc, dset, title  in zip([movData, immData,], \
    [axhm1a, axhm2a], [axetho1a, axetho2a],[axpc1a, axpc2a],\
    [data['AML18_moving'],data['AML18_immobilized'], data['Special_transition']],\
    ['moving (GFP)', 'immobilized (GFP)', 'transient']):
    print key
    # get data
    transient = dset['input'][key]
    results2half = dset['analysis'][key][flag]
    time = transient['Neurons']['TimeFull']
    timeActual = transient['Neurons']['Time']
    #heatmap
#    data1 = transient['Neurons']['RawActivity'][results2half['neuronOrderPCA']]
#    data2 = transient['Neurons']['Ratio'][results2half['neuronOrderPCA']]
#    print np.nanmax(data1), np.nanmax(data2)
#    cax1 = plotHeatmap(timeActual, data1/np.nanmax(data1)-data2/np.nanmax(data2), ax=axhm, vmin=-1, vmax=1)

    cax1 = plotHeatmap(time, transient['Neurons']['ActivityFull'][results2half['neuronOrderPCA']], ax=axhm, vmin=-2, vmax=2)
    axhm.xaxis.label.set_visible(False)
    axhm.set_xticks([])
    axhm.set_title(title)
    #ethogram
    plotEthogram(axetho, time, transient['Behavior']['EthogramFull'], alpha = 1, yValMax=1, yValMin=0, legend=0)
    cleanAxes(axetho, 'all')
    moveAxes(axetho, 'scaley', 0.02)
    axetho.xaxis.label.set_visible(False)
    
    # plot PCA components
    for i in range(np.min([len(results2half['pcaComponents']), 3])):
        #y = results['pcaComponents'][i]
        y = results2half['fullData'][i]
        # normalize
        y =y -np.min(y)
        y =y/np.max(y)
        if key == 'BrainScanner20160506_160928':
            axpc.text(-100, np.mean(y)+i*1.05, 'PC{}'.format(i+1), color = 'k')
        axpc.plot(time[transient['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = 'k')
    
    yloc = axpc.get_ylim()[-1]*1.1
    # labels and such
    axpc.set_xlabel('Time (s)')
    axpc.set_xlim([np.min(timeActual), np.max(timeActual)])
    axpc.set_ylim([axpc.get_ylim()[0], yloc*1.01])
    cleanAxes(axpc, where='y')
    moveAxes(axpc, 'down', 0.02)
## legend for ethogram
#moveAxes(axEthoLeg, 'right', 0.025)
moveAxes(axEthoLeg, 'up', 0.02)
cleanAxes(axEthoLeg, where='all')

#handles, labels = axetho.get_legend_handles_labels()
#leg = mpl.legend.Legend(axEthoLeg, handles[::-1], labels[::-1],frameon=1, loc=1,prop={'size':12},handlelength=0.5, labelspacing=0,handletextpad=0.5)#,bbox_to_anchor=(-1, 0.9), loc=9)
#for hndl in leg.legendHandles:
#    hndl._sizes = [0]
#axEthoLeg.add_artist(leg);
#
# # colorbar
#cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
#cbar.set_ticks([-0.5,0,2])
#cbar.set_ticklabels(['<-0.5',0,'>2'])
#cbar.outline.set_visible(False)
#moveAxes(axcb, 'left', 0.06)
#moveAxes(axcb, 'scaley', -0.08)
#axcb.set_ylabel(r'$\Delta R/R_0$', labelpad = -25)

plt.show()
