

# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure FigS1 - second transition dataset.
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
for typ in ['AML18', 'AML175', 'Special']:
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
fig = plt.figure('S1_Neural dynamics in immobile transitions', figsize=(4.75, 9*2/4.))
gsHeatmap = gridspec.GridSpec(4,2,  width_ratios=[ 1, 0.1], height_ratios = [0.1,1,0.1,0.75])
gsHeatmap.update(left=0.15, right=0.92,  bottom = 0.12, top=0.95, hspace=0.15, wspace=0.25)
fig.patch.set_alpha(0.0)
axTetra =  plt.subplot(gsHeatmap[0,0])

#heatmap axes
axhm1 = plt.subplot(gsHeatmap[1,0])
#axhm2 = plt.subplot(gsHeatmap[0,1])
#axhm3 = plt.subplot(gsHeatmap[0,2])
axcb = plt.subplot(gsHeatmap[1,1])
# ethogram
axetho1 = plt.subplot(gsHeatmap[2,0], clip_on=False)
#axetho2 = plt.subplot(gsHeatmap[1,1], clip_on=False)
#axetho3 = plt.subplot(gsHeatmap[1,2], clip_on=False)
# legend for ethogram
axEthoLeg = plt.subplot(gsHeatmap[2,1])#,clip_on=False)
# principal components
axpc1 =plt.subplot(gsHeatmap[3,0], clip_on=False)#, sharex=axhm)
#axpc2 =plt.subplot(gsHeatmap[2,1], clip_on=False)#, sharex=axhm)
#axpc3 =plt.subplot(gsHeatmap[2,2], clip_on=False)#, sharex=axhm)

################################################
#
# letters
#
################################################

# add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C']
x0 = 0
locations = [(x0,0.95),  (x0,0.5), (x0,0.38)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)

################################################
#
#first row -- show example heatmaps for a movng and immobile dataset
# this is for GFP
################################################
movData = 'BrainScanner20160506_155051'
immData = 'BrainScanner20180518_094052'
transientData = 'BrainScanner20180329_152141'
    
for key, axhm,axetho, axpc, dset, title  in zip([ transientData], \
    [axhm1], [axetho1],[axpc1],\
    [data['Special_transition']],\
    ['transient']):
        
    
    # get data
    transient = dset['input'][key]
    results2half = dset['analysis'][key]['PCAHalf2']
    time = transient['Neurons']['TimeFull']
    timeActual = transient['Neurons']['Time']
    test, train = dset['analysis'][key]['Training']['Half']['Test'], dset['analysis'][key]['Training']['Half']['Train']

    # add tetramisole
    yloc = 1
    axTetra.text(np.mean(timeActual[train[-1]]), 0.94*yloc, "+ paralytic",horizontalalignment='left', color='k', fontsize=fs)
    # the most complicated way to get a step drawn
    axTetra.step([timeActual[train[-1]],timeActual[test[-1]]], [0.92*yloc, 0.92*yloc], color='k', linestyle='-')
    axTetra.plot([timeActual[train[0]],timeActual[train[-1]]], [0.86*yloc, 0.86*yloc], color='k', linestyle='-')
    axTetra.plot([timeActual[train[-1]],timeActual[train[-1]]], [0.86*yloc, 0.92*yloc], color='k', linestyle='-')         
    cleanAxes(axTetra)
    axTetra.set_xlim([np.min(timeActual), np.max(timeActual)])
    
    #heatmap
    cax1 = plotHeatmap(time, transient['Neurons']['ActivityFull'][results2half['neuronOrderPCA']], ax=axhm, vmin=-2, vmax=2)
    axhm.xaxis.label.set_visible(False)
    axhm.set_xticks([])
    #axhm.set_title(title)
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
#        if key == 'BrainScanner20170613_134800':
        axpc.text(-100, np.mean(y)+i*1.05, 'PC{}'.format(i+1), color = 'k')
        axpc.plot(time[transient['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = 'k')
    
    yloc = axpc.get_ylim()[-1]*1.1
    # labels and such
    axpc.set_xlabel('Time (s)')
    axpc.set_xlim([np.min(timeActual), np.max(timeActual)])
    axpc.set_ylim([axpc.get_ylim()[0], yloc*1.01])
    cleanAxes(axpc, where='y')
    moveAxes(axpc, 'down', 0.02)
    
    
    ## indicate immobilization etc
    for label, segment in zip(['moving', 'immobilized'], [train, test]):
    
        axpc.text(np.mean(timeActual[segment]), 1.02*yloc, label,horizontalalignment='center', color=colorsExp[label])
        axpc.plot([timeActual[segment[0]],timeActual[segment[-1]]], [yloc, yloc], color=colorsExp[label])
    # add tetramisole
    #axpc.text(np.mean(timeActual[train[-1]]), 0.98*yloc, "+ tet",horizontalalignment='left', color='k')
    #axpc.plot([timeActual[train[-1]],timeActual[test[-1]]], [0.96*yloc, 0.96*yloc], color='k', linestyle='--')
## legend for ethogram
moveAxes(axEthoLeg, 'right', 0.07)
moveAxes(axEthoLeg, 'up', 0.02)
cleanAxes(axEthoLeg, where='all')

handles, labels = axetho.get_legend_handles_labels()
leg = mpl.legend.Legend(axEthoLeg, handles[::-1], labels[::-1],frameon=1, loc=1,prop={'size':12},handlelength=0.5, labelspacing=0,handletextpad=0.5)#,bbox_to_anchor=(-1, 0.9), loc=9)
for hndl in leg.legendHandles:
    hndl._sizes = [0]
axEthoLeg.add_artist(leg);

 # colorbar
cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
cbar.set_ticks([-2,0,2])
cbar.set_ticklabels(['<-0.5',0,'>2'])
cbar.outline.set_visible(False)
moveAxes(axcb, 'left', 0.06)
moveAxes(axcb, 'scaley', -0.08)
axcb.set_ylabel(r'$\Delta I/I_0$', labelpad = 0, rotation=-90)
plt.show()


