
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 13:15:14 2018
Figure S6 - correlation of neurons with behavior.
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
for typ in ['AML32', 'AML18', 'AML70', 'AML175']:
    for condition in ['chip', 'moving', 'immobilized']:# ['moving', 'immobilized', 'chip']:
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
fig = plt.figure('FigS5NeuronBehaviorCorrelation', figsize=(9.5, 4.))
# this gridspec makes one example plot of a heatmap with its PCA
#gs1 = gridspec.GridSpec(4, 3, width_ratios = [1,1,1], height_ratios=[0.1, 1,1,2])
#gsHeatmap = gridspec.GridSpecFromSubplotSpec(4,5, subplot_spec=gs1[0:4,:], width_ratios=[1.25, 0.1, 0.5,0.5,0.5], height_ratios = [0.1,10,10,10], wspace=0.3, hspace=0.25)
gs1 = gridspec.GridSpec(1,2, width_ratios=[1,1])
gs1.update(left=0.1, right=0.98,  bottom = 0.15, top=0.93, hspace=0.45, wspace=0.25)
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

#letters = ['C', 'D']
#y0 = 0.49
#locations = [(0,y0),  (0.5,y0), (0.62,y0)]
#for letter, loc in zip(letters, locations):
#    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
#            horizontalalignment='left',verticalalignment='top',)

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
## second row
##
#################################################
# variance explained for moving and immobile 
nComp =10
movExp = ['AML32_moving', 'AML70_chip']
imExp = ['AML32_immobilized', 'AML70_immobilized']
movCtrl = ['AML18_moving', 'AML175_moving']
#ax1 = plt.subplot(gs1[0,0])
ax1 = plt.subplot(gs1[0,0])
ax2 = plt.subplot(gs1[0,1])

#ax4 = plt.subplot(gs1[0,3])
nneurons = []
for condition, keys, mark in zip([ 'immobilized','moving'], [ imExp,movExp], ['s', '^']):
    n = []
    for key in keys:
        
        dset = data[key]['analysis']
        tmpdata = []
        
        for idn in dset.keys():
            results=  dset[idn]['PCA']
            
            tmpdata.append(np.cumsum(results['expVariance'][:nComp]*100))
            # for later -- no neurons
            n.append(data[key]['input'][idn]['Neurons']['Activity'].shape[0])            
        #ax1.plot(np.arange(1,nComp+1),np.array(tmpdata).T ,'-',color =colorsExp[condition], lw=1, label = '{} {}'.format(typ, condition),alpha=0.3 )
    #ax1.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = colorsExp[condition], marker=mark)
    nneurons.append(n)
#ax1.set_ylabel('Variance exp. (%)')
#ax1.set_yticks([0,25,50,75,100])
#ax12.set_yticks([0,25,50,75,100])
#ax1.set_xlabel('# of components')

# number of neurons found
#nneurons = np.array(nneurons)
#print nneurons
#mkStyledBoxplot(ax1,[0,1],nneurons, [B1, R1], [ 'immobilized', 'moving'])
#print 'Number of neurons (mean, sem) immobilized', np.mean(nneurons[0]), np.std(nneurons[0])
#print 'Number of neurons (mean, sem) moving:', np.mean(nneurons[1]), np.std(nneurons[1])
#ax1.set_ylabel('# of Neurons')
#ax1.set_xlim([-0.5,1.5])

# neurons correlating with velocity -- make histogram and average
corrV = []
for condition, keys, c in zip([ 'GFP','GCamp6s'], [ movCtrl,movExp], [N0, R1]):
    cv1 = []
    cv2 = []
    mutualInfoV = []
    mutualInfoBC = []
    n = 0
    for key in keys:
        dset = data[key]['analysis']
        tmpdata = []
        for idn in dset.keys():
            # correlation with velocity
            cv1.append(data[key]['analysis'][idn]['Correlation']['AngleVelocity'])
            print key,idn, np.max(cv1[-1])
            ax1.plot( np.sort(cv1[-1])[::-1],1-np.linspace(0,1,len(cv1[-1])), color=c, alpha=0.75, label= condition)
            
            cv2.append(data[key]['analysis'][idn]['Correlation']['Eigenworm3'])
            print key, np.max(cv2[-1])
            ax2.plot( np.sort(cv2[-1])[::-1], 1-np.linspace(0,1,len(cv2[-1])),color=c, alpha=0.75, label= condition)
            n+=1
    corrV.append(cv1)
    print condition, 'N = ', n
##############

##############
ax1.set_xlabel('$R^2$')
ax1.set_ylabel('Fraction of Neurons \n with correlation < $R^2$')
ax2.set_xlabel('$R^2$')
ax1.set_title('Velocity')
ax2.set_title('Body curvature')

###################################
###
### Calculate mutual information between behavior and neural data
###
####################################
#ax3 = plt.subplot(gs1[1,0])
#ax4 = plt.subplot(gs1[1,1])
#nbits = 3
#cnt = 0
#for condition, keys, c in zip([ 'GFP','GCamp6s'], [ movCtrl,movExp], [N0, R1]):
#    mutualInfoV = []
#    mutualInfoBC = []
#    
#    for key in keys:
#        dset = data[key]['input']
#        
#        for idn in dset.keys():
#            Neur = dset[idn]['Neurons']['Activity'].T
#            v =  dset[idn]['Behavior']['AngleVelocity']
#            t = dset[idn]['Behavior']['Eigenworm3']
#            vDig = np.digitize(v, np.linspace(np.min(v), np.max(v), nbits))
#            tDig = np.digitize(t, np.linspace(np.min(t), np.max(t), nbits))
#            
#            
#            # mutual info score turns
#            tmpMIv = mutual_info_classif(Neur, vDig)
#            #ax3.plot( np.sort(tmpMI)[::-1],1-np.linspace(0,1,len(tmpMI)), color=c, alpha=0.75, label= condition)
#            # mutual info score turns
#            tmpMIbc = mutual_info_classif(Neur, tDig)
#            #ax4.plot( np.sort(tmpMI)[::-1],1-np.linspace(0,1,len(tmpMI)), color=c, alpha=0.75, label= condition)
#            mutualInfoV.append([np.median(tmpMIv), np.max(tmpMIv)])
#            mutualInfoBC.append([np.median(tmpMIbc), np.max(tmpMIbc)])
#            print idn, mutualInfoV[-1]
#    cnt += 1
#    # plot 0 = median or 1 = max MI
#    mkStyledBoxplot(ax3, [0+cnt,1+cnt], [np.array(mutualInfoV)[:,1]], [c], ['V'])
#    mkStyledBoxplot(ax4, [0+cnt,1+cnt], [np.array(mutualInfoBC)[:,1]], [c], ['BC'])
#
###############
#ax3.set_xlim([0.5,2.5])
#ax4.set_xlim([0.5,2.5])
#ax3.set_xticks([1,2])
#ax3.set_xticklabels(['GFP', 'GCaMP'], rotation=0)
#ax4.set_xticks([1,2])
#ax4.set_xticklabels(['GFP', 'GCaMP'], rotation = 0)
#ax3.set_xlabel('')
#ax3.set_ylabel('Max(Mutual information)')
#ax4.set_xlabel('')
#ax3.set_title('Velocity')
#ax4.set_title('Body curvature')
plt.show()
##################################
##
## Activity analysis like PNAS paper
##
##################################
#gsAct = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs1[0,:], wspace=0.5)
#ax13 = plt.subplot(gsAct[0,0])
#ax14 = plt.subplot(gsAct[0,1])
#ax15 = plt.subplot(gsAct[0,2])
## extract neural activity histogram for all datasets
#bins = np.linspace(-3,3,30)
#xticks = [-1,0,1,2,3]
##xticks = [-0.1,0.1]
#x = bins[:-1] + np.diff(bins)*0.5
#dx = np.diff(bins)[0]
#activities = []
#gfp = {'moving':['AML18_moving', 'AML175_moving'], 'immobilized':['AML18_immobilized']}
#gcamp = {'moving':['AML32_moving', 'AML70_chip'], 'immobilized':['AML32_immobilized', 'AML70_immobilized']}
#for typ in [gfp, gcamp]:
#    for condition in ['moving', 'immobilized']:
#        keys = typ[condition]
#        tmpdata = []
#        for key in keys:
#            dset = data[key]['input']
#            
#            for idn in dset.keys():
#                X = dset[idn]['Neurons']['Activity']
#               
#                #tmpdata.append(np.mean([np.histogram(n[np.isfinite(n)], bins, density=True)[0] for n in X], axis=0)*dx)
#                tmpdata.append(np.histogram(X[np.isfinite(X)], bins, density=1)[0]*dx)
#        
#        activities.append(tmpdata)
#
#
## plot gfp and gcamp moving in a panel
#histograms = []
#meandata = []
#for hindex, (hist, c) in enumerate(zip([activities[2],  activities[0]], [colorsExp['moving'], colorCtrl['moving']])):
#    m, s = np.nanmean(hist, axis=0), np.nanstd(hist, axis=0)/np.sqrt(len(hist))
#    # normalize now
##    s/=np.sum(m)
##    m /= np.sum(m)
#    #[ax13.plot(x, h, color=c) for h in hist]
#    ax13.plot(x, m, color = c, zorder=2)
#    ax13.fill_between(x, m-s,m+s, color = c, alpha=0.5)
#    histograms.append(m)
#    meandata.append([np.sum(h[x>1]) for h in hist])
#    
## plot gfp and gcamp immobiulized in a panel
#for hindex, (hist, c) in enumerate(zip([activities[3],  activities[1]], [colorsExp['immobilized'], colorCtrl['immobilized']])):
#    m, s = np.nanmean(hist, axis=0), np.nanstd(hist, axis=0)/np.sqrt(len(hist))
#    ax14.plot(x, m, color = c, zorder=2)
#    ax14.fill_between(x, m-s,m+s, color = c, alpha=0.5)
#    histograms.append(m)
#    meandata.append([np.sum(h[x>1]) for h in hist])
#
#ax13.set_xlabel(r'$\Delta R/R_0$')
#ax13.set_xticks(xticks)
#ax14.set_xticks(xticks)
#ax14.set_xlabel(r'$\Delta R/R_0$')
#ax13.set_ylabel('Normalized Counts')
#
## plot probability of signal
#ax15.plot(x, histograms[0]/(histograms[0]+histograms[1]), color=colorsExp['moving'])
#ax15.plot(x, histograms[2]/(histograms[2]+histograms[3]), color=colorsExp['immobilized'])
## draw a line where 95 percent level
#ax15.axhline(y = 0.75,color='k', linestyle = '--')
#ax15.set_xlabel(r'$\Delta R/R_0$')
#ax15.set_ylabel('P(Signal)')
#ax15.set_xticks(xticks)
#ax15.set_yticks([0.5,1])
#
#print [np.sum(h[x>1]) for h in histograms]
#print [np.sum(h) for h in histograms]
## boxplot of signal percentage in each recording
#ax12 = plt.subplot(gsAct[3])
#color, labels, ydata = [],[],[]
#
##gfp = {'moving':['AML18_moving', 'AML175_moving'], 'immobilized':['AML18_immobilized']}
##gcamp = {'moving':['AML32_moving', 'AML70_chip'], 'immobilized':['AML32_immobilized', 'AML70_immobilized']}
##for typ in [gfp, gcamp]:
##    for condition in ['moving', 'immobilized']:
##        keys = typ[condition]
##        tmpdata = []
##        for key in keys:
##            dset = data[key]['input']
##            
##            for idn in dset.keys():
##                print np.nanstd(dset[idn]['Neurons']['Activity'], axis=1).shape
##                tmpdata.append(np.nanmean(np.nanmean(dset[idn]['Neurons']['Activity'], axis=1)))
##        ydata.append(tmpdata)
#
#color = [N0, N1, R1, B1]
#labels = ['M(Ctrl)', 'I(Ctrl)', 'M', 'I']
#color = [R1, N0, B1, N1]
#labels = ['M', 'M(Ctrl)', 'I','I(Ctrl)']
#
#ydata = np.array(meandata).T
#x_data = np.arange(len(ydata))
#mkStyledBoxplot(ax12, x_data, ydata, color, labels)
#ax12.set_xlim([-0.5, 3.25])
#ax12.set_ylabel(r'$P(\Delta R/R_0)>1$')
plt.show()
        #ax.imshow( tmpdata, aspect='auto', interpolation='none', origin='lower',extent=[T[0],T[-1],len(tmpdata),0],vmax=1)
#ax13.text(-0.25,0,'Autocorrelation', fontsize=14,transform = ax13.transAxes, rotation=90, verticalalignment ='center')

        
#ax11.set_ylabel('Explained variance (%)')
#ax11.set_yticks([0,25,50,75,100])
#ax12.set_yticks([0,25,50,75,100])
#ax11.text(1.2,-0.3,'Number of components', transform = ax11.transAxes, horizontalalignment ='center')



#gsEWs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gsHeatmap[3,0], wspace=0.1, hspace=0.2, height_ratios=[1,1,1])
## plot eigenworm schematic
#ax6 = plt.subplot(gsEWs[2,0],  adjustable='box')
#ax6.axis('equal')
## plot eigenworms
#eigenworms = dh.loadEigenBasis(filename = 'utility/Eigenworms.dat', nComp=3, new=True)
#lengths = np.array([4])
#meanAngle = np.array([-0.07945])
#refPoint = np.array([[0,0]])
#prefactor = [r'',r'$=a_1$ *',r'+ $a_2$ *',r'+ $a_3$ *' ][::-1]
#descr = ['posture', 'undulation', 'undulation', 'turn'][::-1]
#colors = [B1, B1, R1]
## from which volume to show the posture
#tindex = 500
#for i in range(4):
#    pcs = np.zeros(3)
#    if i ==3:
#        pcsNew, meanAngle, _, refPoint = dh.calculateEigenwormsFromCL(moving['CL'], eigenworms)
#        pc3New, pc2New, pc1New = pcsNew
#        cl = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
#        
#        new = createWorm(cl[tindex,:,0], cl[tindex,:,1])
#    else:
#        pcs[i] = 10
#        clNew = dh.calculateCLfromEW(pcs, eigenworms[:3], meanAngle, lengths, refPoint)[0]
#        new = createWorm(clNew[:,0], clNew[:,1])
#    
#    new -=np.mean(new, axis=0)
#    new[:,0] -= i*600 - 800
#    x, y = np.mean(new, axis =0)
#    p2 = mpl.patches.Polygon(new, closed=True, fc=N0, ec='none')
#    plt.text(x,y-200, descr[i], fontsize=12, horizontalalignment = 'center')
#    plt.text(x-250,y, prefactor[i], fontsize=12, horizontalalignment = 'center')
#    ax6.add_patch(p2)
#    ax6.axes.get_xaxis().set_visible(False)
#    ax6.axes.get_yaxis().set_visible(False)
#    ax6.spines['left'].set_visible(False)
#    ax6.spines['bottom'].set_visible(False)
#    ax6.set_xlim(-1200, 1000)
#    ax6.set_ylim(-50, 10)
#
#
## plot angle velocity and turns
#ax7 = plt.subplot(gsEWs[0,0],sharex=axhm )
#ax7.plot(timeActual, moving['Behavior']['AngleVelocity'], color = R1)
## draw a box for the testset
#ax7.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
#ax7.axhline(color='k', linestyle = '--', zorder=-1)
#ax7.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
#ax7.set_ylabel('Wave speed')
## make scalebar
#xscale = timeActual[0]-20
#yscale =  [-0.025, 0.025]
#ax7.plot([xscale, xscale], yscale, color=R1, clip_on=False)
#ax7.text(xscale, np.max(ax7.get_ylim())*1.1, 'Wave speed', color=R1,horizontalalignment='center',verticalalignment='center')
#ax7.text(xscale, 0, np.ptp(yscale), color=R1, rotation = 90,horizontalalignment='right',verticalalignment='center')
#ax7.axes.get_yaxis().set_visible(False)
#ax7.spines['left'].set_visible(False)
## remove xlabels
#plt.setp(ax7.get_xticklabels(), visible=False)
#ax8 = plt.subplot(gsEWs[1,0], sharex=axhm)
#ax8.plot(timeActual,moving['Behavior']['Eigenworm3'], color = B1)
## draw a box for the testset
#ax8.axvspan(timeActual[test[0]], timeActual[test[-1]], color=N2, zorder=-10, alpha=0.75)
##ax8.text(np.mean(timeActual[test]), ax4.get_ylim()[-1], 'Testset',horizontalalignment='center')
#ax8.axhline(color='k', linestyle ='--', zorder=-1)
#ax8.set_ylabel('Turn')
#ax8.set_xlabel('Time (s)')
#ax8.set_xlim([timeActual[0], timeActual[-1]])
## make scalebar
#xscale = timeActual[0]-20
#yscale =  [-7.5, 7.5]
#ax8.plot([xscale, xscale], yscale, color=B1, clip_on=False)
#ax8.text(xscale, np.max(ax8.get_ylim()), 'Turns', color=B1,horizontalalignment='center',verticalalignment='center')
#ax8.text(xscale, 0, int(np.round(np.ptp(yscale))), color=B1, rotation = 90,horizontalalignment='right',verticalalignment='center')
#ax8.axes.get_yaxis().set_visible(False)
#ax8.spines['left'].set_visible(False)
## move up all of them
#ax6.set_zorder(-10)
#for axtmp in [ax7, ax8]:
#    pos = axtmp.get_position().get_points()
#    pos[:,1] +=0.01
#    posNew = mpl.transforms.Bbox(pos)
#    axtmp.set_position(posNew)
#     
#pos=ax6.get_position().get_points()
#posNew = mpl.transforms.Bbox(pos)
#ax6.set_position(posNew)
#moveAxes(ax6, action='scale', step=0.05 )
#moveAxes(ax6, action='down', step=0.02 )
## schematic of behavior prediction from PCA
#gsScheme = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gsHeatmap[3,1:],width_ratios=[1.2,1], wspace=0.1, hspace=0.4)
#axscheme1 = plt.subplot(gsScheme[0,0])
#axscheme1.axis('equal')
#axscheme2 = plt.subplot(gsScheme[0,1], sharey=axscheme1)
## input are PCs, let's only show the testset
##
#x,y,z = movingAnalysis['PCA']['pcaComponents'][:3, test]
#t = moving['Neurons']['Time'][test]
#scale = np.ptp(t)*0.8
#ylocs = np.linspace(0,scale,3)
## for weight circles
#ycircle1 = ylocs +scale/12.
#ycircle2 = ylocs -scale/12.
#xcirc = t[-1]+scale/7
#for lindex, line in enumerate([x,y,z]):
#    line -=np.mean(line)
#    line /=np.max(line)
#    line*=scale/7.
#    axscheme1.plot(t,line+ylocs[lindex], color='k')
#    axscheme1.text(t[0], ylocs[lindex]+scale/7, 'PC{}'.format(lindex+1), horizontalalignment='center')
#    # circles for weights - red
#    circ = mpl.patches.Circle((xcirc, ycircle1[lindex]), scale/10.,fill=True,color='w',lw=2, ec=R1, clip_on=False)
#    axscheme1.text(xcirc, ycircle1[lindex], r'$w_{}$'.format(lindex+1),color=R1, verticalalignment='center', horizontalalignment='center')
#    axscheme1.add_patch(circ)
#    #blue circles
#    circ = mpl.patches.Circle((xcirc, ycircle2[lindex]), scale/10.,fill=True,linestyle=(0, (1, 1)),color='w',lw=2, ec=B1, clip_on=False)
#    axscheme1.text(xcirc, ycircle2[lindex], r'$w_{}$'.format(lindex+1),color=B1, verticalalignment='center', horizontalalignment='center')
#    axscheme1.add_patch(circ)
#    
#ybeh = [ylocs[1]+scale/10., ylocs[0]-scale/10.]+np.diff(ylocs)/2.
#for behavior, color, cpred, yl, label in zip(['AngleVelocity','Eigenworm3' ], \
#            [N1, N1], [R1, B1], ybeh, ['Wave speed', 'Turn']):
#    beh = moving['Behavior'][behavior][test]
#    meanb, maxb = np.mean(beh),np.std(beh)
#    beh = (beh-meanb)/maxb
#    beh*=scale/10
#    behPred = movingAnalysis['PCAPred'][behavior]['output'][test]
##    behPred = (behPred-meanb)/maxb
#    behPred*=scale/10
#    axscheme2.plot(t, beh+yl, color=color)
#    axscheme2.plot(t, behPred+yl, color=cpred)
#    axscheme2.text(t[-1], yl+scale/5, \
#    r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis['PCAPred'][behavior]['scorepredicted'])), horizontalalignment = 'right')
#    axscheme2.text(t[-1]*1.1, yl, label, rotation=90, color=cpred, verticalalignment='center')
##axscheme2.set_zorder(-1)
#
#axscheme2.set_facecolor('none')
#for i in range(3):
#    con = mpl.patches.ConnectionPatch(xyA=(xcirc,ycircle1[i]), xyB=(t[0],ybeh[0]), coordsA="data", coordsB="data",
#                          axesA=axscheme1, axesB=axscheme2, color=R1)
#    axscheme1.add_artist(con)
#    con.set_zorder(-10)    
#    con = mpl.patches.ConnectionPatch(xyA=(xcirc,ycircle2[i]), xyB=(t[0], ybeh[1]), coordsA="data", coordsB="data",
#                          axesA=axscheme1, axesB=axscheme2, color=B1, lw=2, linestyle=':')
#    axscheme1.add_artist(con)
#    con.set_zorder(-10)
## add scalebar
#l =120
#y = ylocs[0] - scale/4.
#axscheme1.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
#axscheme1.text(t[0]+l*0.5,y*0.8, '2 min', horizontalalignment='center')
#axscheme2.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
#axscheme2.text(t[0]+l*0.5,y*0.8, '2 min', horizontalalignment='center')
#
#for axtmp in [axscheme1, axscheme2]:
#    axtmp.spines['left'].set_visible(False)
#    axtmp.spines['bottom'].set_visible(False)
#    axtmp.set_yticks([])
#    axtmp.set_xticks([])
#    #axtmp.set_xlabel('Time (s)')
#
#axscheme1.set_xlim([t[0], 500])
#
#################################################
##
## fourth row
##
#################################################
## predict neural activity from behavior
#gsLasso = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs1[3,0], width_ratios=[1,1], wspace=0.2, hspace=0.1)
#flag = 'PCAPred'
#
#axNav= plt.subplot(gsLasso[0,0])
#axNt = plt.subplot(gsLasso[0,1])
#
#gsBrokenAxis = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsLasso[0,0])
#
#for behavior, colors, axR2 in zip(['AngleVelocity', 'Eigenworm3'], [(R2, N0), (B2, N0)], [axNav, axNt ]):
#
#    alldata = []
#    # experiment
#    c = colors[0]
#    for key, marker in zip(['AML32_moving', 'AML70_chip'],['o', "^"]):
#        dset = data[key]['analysis']
#        keep = []
#        for idn in dset.keys():
#            results=  dset[idn][flag][behavior]
#            keep.append(results['scorepredicted'])
#            
#        keep = np.array(keep)
#        rnd1 = np.random.rand(len(keep))*0.2
#        axR2.scatter(np.zeros(len(keep))+rnd1, keep, marker = marker, c = c, edgecolor=c, alpha=0.5)
#        alldata.append(keep)
#    alldata = np.array(alldata)
#    mkStyledBoxplot(axR2, [-0.5, 1.5], alldata.T, [c], ['GCamp6s'], scatter=False)
#    # controls
#    c = colors[1]
#    ctrldata = []
#    xoff = 1.5
#    for key, marker in zip(['AML18_moving', 'AML175_moving'],['o', "p"]):
#        dset = data[key]['analysis']
#        keep = []
#        for idn in dset.keys():
#            results=  dset[idn][flag][behavior]
#            keep.append(results['scorepredicted'])
#        keep = np.array(keep)
#        rnd1 = np.random.rand(len(keep))*0.2
#        axR2.scatter(xoff+np.zeros(len(keep))+rnd1, keep, marker = marker,c = c, edgecolor=c, alpha=0.5)
#        ctrldata.append(keep)
#    ctrldata = np.array(ctrldata)
#    mkStyledBoxplot(axR2, [-0.5+xoff, 1.5+xoff], ctrldata.T, [c,], ['Control (GFP)'], scatter=False)
#    
#    axR2.set_xlim([-1, 2.5])
#    axR2.set_xticks([-0.5,-0.5+xoff])
#    axR2.set_xticklabels(['GCaMP6s', 'GFP'])
#axNav.set_ylabel(r'$R^2$ (Testset)')

plt.show()
# get all the weights for the different samples

#for typ, colors, ax in zip(['AML32', 'AML18'], [colorsExp, colorCtrl], [ax11, ax12]):
#    for condition in ['moving', 'immobilized']:
#        key = '{}_{}'.format(typ, condition)
#        dset = data[key]['analysis']
#        tmpdata = []
#        for idn in dset.keys():
#            results=  dset[idn]['PCA']
#            rescale=  data[key]['input'][idn]['Neurons']['Activity'].shape[0]
#            tmpdata.append(np.cumsum(results['expVariance'][:nComp])*100)       
#        ax.plot(np.arange(1,nComp+1),np.mean(tmpdata, axis=0) ,'o-',color = colors[condition], lw=1, label = '{} {}'.format(typ, condition))
#        ax.errorbar(np.arange(1,nComp+1), np.mean(tmpdata, axis=0), np.std(tmpdata, axis=0), color = colors[condition])


#ax10 = plt.subplot(gsPred[0,1])
#for li,line in enumerate(orderedWeights):
#    ax10.plot(np.abs(line), label = ('weights for PC{}'.format(li+1)), color='C5', alpha=0.25+0.05*li, lw=1)
#ax10.set_ylabel('Weights')
#
#ax10.plot(np.mean(np.abs(orderedWeights), axis=0), color='C5', alpha=1, lw=2, marker = 'o')
#ax10.set_xticks(np.arange(len(res['behaviorOrder'])))
#ax10.set_xticklabels(res['behaviorLabels'][res['behaviorOrder']])
#
#plt.setp(ax10.get_xticklabels(), rotation=-25)

### plot correlation of PC axes and velocity/turns
#ax9 = plt.subplot(gsHeatmap[3,2])
#color, labels, ydata1, ydata2 = [],[],[], []
#condition = 'moving'
#
#for typ, colors in zip(['AML32', 'AML18'], [colorsExp, colorCtrl]):
#    
#        color.append(colors[condition])
#        labels.append('{} {}'.format(typ, condition))
#        tmpdata1 = []
#        tmpdata2 = []
#        key = '{}_{}'.format(typ, condition)
#        dset = data[key]['analysis']
#        for idn in dset.keys():
#            tmpdata1.append(dset[idn]['PCACorrelation']['AngleVelocity'][:3])
#            tmpdata2.append(dset[idn]['PCACorrelation']['Eigenworm3'][:3])
#        ydata1.append(tmpdata1)
#        ydata2.append(tmpdata2)
#x_data = np.arange(len(ydata1))
#print ydata1
#mp.mkStyledBoxplot(ax9, x_data, ydata1, color, labels)
#mp.mkStyledBoxplot(ax9, x_data+0.5, ydata2, color, labels)
#plt.show()
#reverse prediction


# plot stuff
#plt.figure('PredictedNeuralActivity', figsize=(2.28*4,2.28*6))
#
## show reduced dimensionality heatmap
#mp.plotHeatmap(moving['Neurons']['Time'][test], res['lowDimNeuro'][:,test])
#plt.subplot(322)
#mp.plotHeatmap(moving['Neurons']['Time'][test], newHM[:,test], vmin=np.min(newHM)*1.1, vmax=np.max(newHM)*0.9)
#plt.subplot(324)
#for ind, i in enumerate(res['PCA_indices'][:4]):
#    x = moving['Neurons']['Time'][test]
#    line1, = plt.plot(x, res['NeuralPCS'][test,i]+ind*12, color='C0', label='Neural PCs')
#    line2, = plt.plot(x, res['predictedNeuralPCS'][test,i]+ind*12, color='C3', label= 'Predicted')
#    plt.text(x[-1]*0.9, 1.2*np.max(res['predictedNeuralPCS'][test,i]+ind*10), '$R^2={:.2f}$'.format(res['R2_test'][ind]))
#plt.legend([line1, line2], ['Neural PCs', 'Predicted from Behavior'], loc=2)
#ylabels = ['PC {}'.format(index+1) for index in res['PCA_indices'][:4]]
#plt.yticks(np.arange(0,4*12, 12), ylabels)
#plt.xlabel('Time(s)')
#plt.subplot(323)
#for ind, i in enumerate(res['behaviorOrder']):
#    plt.plot(moving['Neurons']['Time'], res['behavior'][:,i]+ind*4, color='k', label = res['behaviorLabels'][i], alpha=0.35+0.1*ind)
#    plt.xlabel('Time(s)')
#    
#locs, labels = plt.yticks()
#plt.yticks(np.arange(0,len(res['behaviorOrder'])*4,4), res['behaviorLabels'][res['behaviorOrder']])
##plt.legend()
#plt.subplot(325)
## plot the weights for each PC
#
#for li,line in enumerate(orderedWeights):
#    plt.plot(np.abs(line), label = ('weights for PC{}'.format(li+1)), color='C5', alpha=0.25+0.05*li, lw=1)
#plt.ylabel('Weights')
##plt.xlabel('behaviors')
#plt.plot(np.mean(np.abs(orderedWeights), axis=0), color='C5', alpha=1, lw=2, marker = 'o')
#plt.xticks(np.arange(len(res['behaviorOrder'])), res['behaviorLabels'][res['behaviorOrder']], rotation=30)
#plt.subplot(326)
#
#plt.plot(res['expVariance'], color='C7', alpha=1, lw=2, marker = 'o')
#plt.xticks(np.arange(len(res['behaviorOrder'])),res['behaviorLabels'][res['behaviorOrder']], rotation=30)
#plt.show()
