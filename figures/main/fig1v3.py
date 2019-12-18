
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
#mpl.rcParams["xtick.labelsize"] =18
#import matplotlib.animation as animation
#import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
#import matplotlib.ticker as mtick

#import singlePanels as sp
#import makePlots as mp
import prediction.dataHandler as dh
from prediction import userTracker

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

fig = plt.figure('Fig1Neural dynamics in freely moving animals', figsize=(9.5, 6.5))
gsHeatmap = gridspec.GridSpec(6,4,  width_ratios=[1.5, 0.1, 0.5, 0.5], height_ratios = [0.05,0.95,0.1,0.75,0.05, 0.75])
gsHeatmap.update(left=0.06, right=0.98,  bottom = 0.06, top=0.98, hspace=0.3, wspace=0.75)
#fig.patch.set_alpha(0.0)
# drug treatment axis
axTetra = plt.subplot(gsHeatmap[0,0], clip_on=False)
moveAxes(axTetra, 'down', 0.025)

#heatmap axes
axhm = plt.subplot(gsHeatmap[1,0])
axcb = plt.subplot(gsHeatmap[1,1])
# ethogram
axetho = plt.subplot(gsHeatmap[2,0], clip_on=False)
# legend for ethogram
axEthoLeg = fig.add_axes([0.4,0.42,0.12,0.2])#plt.subplot(gsHeatmap[1:2,1:])#,clip_on=False)
#moveAxes(axEthoLeg, 'up', 0.06)
cleanAxes(axEthoLeg, where='all')
# principal components
ax4 =plt.subplot(gsHeatmap[3:4,0], clip_on=False)#, sharex=axhm)
moveAxes(ax4, 'down', 0.02)
# subpanel layout for autocorr
gsPer= gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsHeatmap[:2,2:], wspace=0.35,hspace=0.2)#, height_ratios=[1,1.25])
ax13 = plt.subplot(gsPer[0])
ax14 = plt.subplot(gsPer[1])
# manifooolds - now 4
gsPer1= gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gsHeatmap[4:,:], wspace=0.35,hspace=0.2)#, width_ratios=[1,1,0.1,1,1])#, height_ratios=[1,1.25])

ax5 = plt.subplot(gsPer1[0], projection='3d', clip_on = False, zorder=-10)# aspect='equal')
ax6 = plt.subplot(gsPer1[1], projection='3d', clip_on = False, zorder=-10)#, aspect='equal')
ax7 = plt.subplot(gsPer1[2], projection='3d', clip_on = False, zorder=-10)#, aspect='equal')
ax8 = plt.subplot(gsPer1[3], projection='3d', clip_on = False, zorder=-10)#, aspect='equal')


axexpV = fig.add_axes([0.48,0.4,0.12,0.2])


gsHM2 =  gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsHeatmap[3,2:], width_ratios=[10,1],wspace=0.35,hspace=0.2)#, height_ratios=[1,1.25])
axhm2 = plt.subplot(gsHM2[0])
axcb2 = plt.subplot(gsHM2[1])


################################################
#
# letters
#
################################################
# mark locations on the figure to get good guess for a,b,c locs
#for y in np.arange(0,1.1,0.1):
#    plt.figtext(0, y, y)
#for x in np.arange(0,1.1,0.1):
#    plt.figtext(x, 0.95, x)
##
#letters = map(chr, range(65, 91)) 
## add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C']
x0 = 0
locations = [(x0,0.95),  (x0,0.6), (x0,0.55)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
#
letters = ['D']
x0 = 0.52
locations = [(x0,0.95)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)

letters = ['E', 'F']
y0 = 0.26
locations = [(0.4,0.55), (0.0,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)

letters = ['G', 'H']
y0 = 0.26
locations = [(0.58,0.55),  (0.52,y0), (0.77,y0)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)
#
#letters = ['I', 'J']
#y0 = 0.27
#locations = [(0,y0),  (0.22,y0), (0.76,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)
################################################
#
#first row
#
################################################
# select a special dataset - transiently immobilized
transientData = 'BrainScanner20180511_134913'
transient = data['Special_transition']['input']['BrainScanner20180511_134913']
transientAnalysis = data['Special_transition']['analysis']['BrainScanner20180511_134913']
# time first half, second half. Indices of times
timeHalf = np.arange(0, 1400)
time2Half = np.arange(1600, transient['Neurons']['Activity'].shape[1])
# pull out repeated stuff
time = transient['Neurons']['TimeFull']
timeActual = transient['Neurons']['Time']
noNeurons = transient['Neurons']['Activity'].shape[0]
results = transientAnalysis['PCA']
resultshalf = transientAnalysis['PCAHalf1']
results2half = transientAnalysis['PCAHalf2']
test, train = transientAnalysis['Training']['Half']['Test'], transientAnalysis['Training']['Half']['Train']
colorsExp = {'moving': R1, 'immobilized': B1}
colorsCtrl = {'moving': N0,'immobilized': N1}
         
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
# colorbar
cbar = fig.colorbar(cax1, cax=axcb, use_gridspec = True)
cbar.set_ticks([-2,0,2])
cbar.set_ticklabels(['<2',0,'>2'])
cbar.outline.set_visible(False)
moveAxes(axcb, 'left', 0.06)
moveAxes(axcb, 'scaley', -0.08)
axcb.set_ylabel(r'$\Delta I/I_0$', labelpad = 10, rotation=-90)
#ethogram

plotEthogram(axetho, time, transient['Behavior']['EthogramFull'], alpha = 1, yValMax=1, yValMin=0, legend=0)
cleanAxes(axetho, 'all')
moveAxes(axetho, 'scaley', 0.02)
axetho.xaxis.label.set_visible(False)
# legend for ethogram
moveAxes(axEthoLeg, 'up', 0.06)

handles, labels = axetho.get_legend_handles_labels()
leg = mpl.legend.Legend(axEthoLeg, handles[::-1], labels[::-1],frameon=1, loc=2,prop={'size':12},ncol=2,handlelength=0.5, labelspacing=0,handletextpad=0.25)#,bbox_to_anchor=(-1, 0.9), loc=9)
for hndl in leg.legendHandles:
    hndl._sizes = [0]
axEthoLeg.add_artist(leg);

# plot mean autocorrelation moving versus immobile
movExp = ['AML32_moving', 'AML70_chip']
imExp = ['AML32_immobilized', 'AML70_immobilized']

# all moving gfp
movCtrl = ['AML18_moving', 'AML175_moving']
imCtrl = ['AML18_immobilized']
# color coding
colorsExp = {'moving': R1, 'immobilized': B1}
colorCtrl = {'moving': N0,'immobilized': N1}

typ, colors, axes = [movExp, imExp], colorsExp, [ax13, ax14]
for  condition, ax, keys in zip(['moving', 'immobilized'], axes, typ):
    tmpdata = []
    for key in keys:
        #key = '{}_{}'.format(typ, condition)
        dset = data[key]['analysis']
        for idn in dset.keys():
            tmpdata.append(np.mean(dset[idn]['Period']['NeuronACorr'],axis=0)) 
            T = dset[idn]['Period']['Periods']
    m, s = np.nanmean(tmpdata, axis=0), np.nanstd(tmpdata, axis=0)
    tmpdata= np.array(tmpdata)
    ax.plot(T,tmpdata.T ,'-',color = colors[condition], lw=1.5, alpha=0.35,label = '{} {}'.format(typ, condition))
    print 'condition', condition, 'N=', len(tmpdata)
    #if typ==imExp:
    #    ax.plot(T,tmpdata[0] ,'-',color = colors[condition], lw=2, alpha=1,label = '{} {}'.format(typ, condition))
    #ax.plot(T,np.mean(tmpdata,axis=0) ,'-',color = colors[condition], lw=5, alpha=0.5,label = '{} {}'.format(typ, condition))
    #ax.fill_between(dset[idn]['Period']['Periods'], m-s, m+s, alpha=0.5, zorder=-1,color = colors[condition])
    ax.axhline(color='k', linestyle = '--', zorder=-1)
    ax.set_ylim([-0.2,1])
    ax.text(0.5, 0.9,condition, transform=ax.transAxes, horizontalalignment='center')
    ax.set_xticks([0,150,300])
    ax.set_yticks([0,0.5,1])
ax13.set_ylabel('Autocorrelation')
#ax13.text(-0.5,0,'Autocorrelation', fontsize=14,transform = ax13.transAxes, rotation=90, verticalment ='center')
ax13.set_xlabel('Lag (s)')
ax14.set_xlabel('Lag (s)')
ax14.set_yticks([])
#ax16.set_yticks([])

# plot autocorr of our favorite dataset
ax13.plot(transientAnalysis['Period1Half']['Periods'],np.mean(transientAnalysis['Period1Half']['NeuronACorr'],axis=0), colorsExp['moving'], lw=1.5)
ax14.plot(transientAnalysis['Period2Half']['Periods'],np.mean(transientAnalysis['Period2Half']['NeuronACorr'],axis=0), colorsExp['immobilized'], lw=1.5)

# plot dimensionality for inactive and active plus together
nComp = 10#results['nComp']
for y, col, lab, mark in zip([resultshalf['expVariance'][:nComp], results2half['expVariance'][:nComp]]\
        , [ R1, B1], [ 'Moving', 'Paralyzed'], [ '^', 's']):
    #ax3.fill_between(np.arange(0.5,nComp+0.5),y*100, step='post', color=col, alpha=0.5)
   axexpV.plot(np.arange(1,nComp+1),np.cumsum(y)*100, 'o-',color = col, label = lab, marker=mark) 
   print 'Variance explained', lab, '3 comps', np.cumsum(y)[2]*100
axexpV.set_ylabel('Variance exp.(%)', labelpad=0)
axexpV.set_yticks([0,25,50,75,100])
axexpV.set_xlabel('# of components')
axexpV.set_xticks([0,5, 10])


#################################################
##
## last row: manifolds
##
#################################################

# plot PCA components
for i in range(np.min([len(results2half['pcaComponents']), 3])):
    #y = results['pcaComponents'][i]
    y = results2half['fullData'][i]
    # normalize
    y =y -np.min(y)
    y =y/np.max(y)
    ax4.text(-100, np.mean(y)+i*1.05, r'PC$_{}$'.format(i+1), color = 'k')
    ax4.plot(time[transient['Neurons']['valid']], i*1.1+y, label='Component {}'.format(i+1), lw=1, color = 'k')

yloc = ax4.get_ylim()[-1]*1.05
## indicate immobilization etc
for label, segment in zip(['moving', 'immobilized'], [train, test]):
    ax4.text(np.mean(timeActual[segment]), 1.02*yloc, label,horizontalalignment='center', color=colorsExp[label], fontsize=fs)
    ax4.plot([timeActual[segment[0]],timeActual[segment[-1]]], [yloc, yloc], color=colorsExp[label])

# labels and such
ax4.set_xlabel('Time (s)')
ax4.set_xlim([np.min(timeActual), np.max(timeActual)])
ax4.set_ylim([ax4.get_ylim()[0], yloc*1.01])
cleanAxes(ax4, where='y')
#
#############################################
# plot manifold! MANIFOOOOOLD!
############################################
# plot manifold for split dataset
x,y,z = results2half['fullData'][:3]
x/=np.max(x)
y/=np.max(y)
z/=np.max(z)

# make smoooth
smooth = 12
x = gaussian_filter1d(x, smooth)
y = gaussian_filter1d(y, smooth)
z = gaussian_filter1d(z, smooth)
# color by before and after
colorBy = np.zeros(len(timeActual))
colorBy[train] = 1 # first four minutes is m9
ax5.plot(x[train],y[train],z[train], color=R1)
ax6.plot(x[test],y[test],z[test], color=B1)
ax5.scatter(x[train[::12]],y[train[::12]],z[train[::12]], color=R1, s=5)
ax6.scatter(x[test[::12]],y[test[::12]],z[test[::12]], color=B1, s=5)

def juggleAxes(ax, extent = [-0.75,0.75,-0.75,0.75,-1,1]):
    """change some axes order etc."""
   
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    
#        #move axes
    axesNames = [ax.xaxis, ax.yaxis, ax.zaxis]
    for tmp, loc in zip(axesNames, [(0,0,0),(1,1,1),(2,2,2)]):
        tmp._axinfo['juggled']=loc
    ax.set_xlim([extent[0], extent[1]])
    ax.set_ylim([extent[2], extent[3]])
    ax.set_zlim([extent[4], extent[5]])
    moveAxes(ax, action='scale', step=0.05)
    moveAxes(ax, action='down', step=0.045)
    moveAxes(ax, action='scalex', step=0.025)

def scalebar(ax, center = (-1.25,0.5,-1.)):
    #        # make a scale bar in 3d
    scX, scY, scZ = center
    names = [r'PC$_1$', 'PC$_2$', 'PC$_3$']
    align = ['center', 'right','center']
    aligny = [ 'top','center','baseline','center']
    for i in range(3):
        l = np.zeros(3)
        l[i] = 0.25
        ax.plot([scX, scX -l[0]], [scY, scY+l[1]], [scZ, scZ+l[2]], color='k', clip_on=False)
        ax.text(scX -l[0], scY+l[1], scZ+l[2], names[i],color='k',fontsize=12, \
        horizontalalignment=align[i], verticalalignment=aligny[i])
    #l = np.zeros(3)#+axmin


for ax, label  in zip([ax5, ax6], ['moving', 'immobilized']):    
    ax.view_init(elev=32, azim=-116)
    #ax.set_title(label, fontsize=mpl.rcParams['font.size'])
#        ax.dist = 7
    juggleAxes(ax)
    moveAxes(ax, 'left', 0.05)
scalebar(ax6)


##########################
# heatmap -- fully immobile worm
###############################
immWorm = 'BrainScanner20171017_184114'
heatData = data['AML32_immobilized']['input'][immWorm]['Neurons']['ActivityFull']
order = data['AML32_immobilized']['analysis'][immWorm]['PCA']['neuronOrderPCA']
#heatmap
cax1 = plotHeatmap(time, heatData[order], ax=axhm2, vmin=-2, vmax=2)

axhm2.set_xlabel('Time (s)')
#axhm2.set_xticks([])
# colorbar
cbar = fig.colorbar(cax1, cax=axcb2, use_gridspec = True)
cbar.set_ticks([-2,0,2])
cbar.set_ticklabels(['<2',0,'>2'])
cbar.outline.set_visible(False)
moveAxes(axcb2, 'left', 0.04)
moveAxes(axcb2, 'scaley', -0.08)
axcb2.set_ylabel(r'$\Delta I/I_0$', labelpad = 10, rotation=-90)
##########################
# second manifold -- fully immobile worm
###############################
    
    
dset = data['AML32_immobilized']['analysis'][immWorm]
# pull out all the components
x, y, z = dset['PCA']['fullData'][:3]

# normalize components
x/=np.max(x)
y/=np.max(y)
z/=np.max(z)
# smooth
# make smoooth
smooth = 12

x = gaussian_filter1d(x, smooth)
y = gaussian_filter1d(y, smooth)
z = gaussian_filter1d(z, smooth)
# plot in 3d

ax7.plot(x, y, z, color=N1, zorder=-10)
ax7.scatter(x[::12],y[::12],z[::12], color=B1, s=2)
ax8.view_init(elev=40, azim=150)
ax7.set_xlabel(r'PC$_1$', labelpad = 5)
ax7.set_ylabel(r'PC$_2$', labelpad = 5)
ax7.set_zlabel(r'PC$_3$', labelpad = 5)

ax8.plot(x, y, z, color=N1, zorder=-10)
ax8.scatter(x[::12],y[::12],z[::12], color=B1, s=2)
ax8.view_init(elev=-5, azim=-79)

ax8.set_xlabel(r'PC$_1$', labelpad = 5)
ax8.set_ylabel(r'PC$_2$', labelpad = 0)
ax8.set_zlabel(r'PC$_3$', labelpad = 5)
juggleAxes(ax7)
juggleAxes(ax8)
#scalebar(ax7, center= (1.2, 0.3 , -1.25))
#scalebar(ax8, center= (0.5, 0.5 , -1.5))
#ax7.set_title('View 1')
#ax8.set_title('View 2')
# some movement for axes
for ax in [axexpV]:
    alignAxes(ax4, ax, where='y')
moveAxes(axexpV, 'scale', -0.05)
moveAxes(axexpV, 'up', 0.025)
moveAxes(axexpV, 'right', -0.01)

moveAxes(ax13, 'scaley', -0.03)
moveAxes(ax14, 'scaley', -0.03)
moveAxes(ax8, 'down', 0.02)
plt.show()

