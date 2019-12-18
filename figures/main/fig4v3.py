
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
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from scipy.special import erf 
# custom pip
from scipy.stats import pearsonr
#

import prediction.dataHandler as dh
from prediction import userTracker
from prediction.stylesheet import *
# stats
from sklearn.metrics import r2_score,mean_squared_error
from scipy.interpolate import UnivariateSpline
from scipy import ndimage

# suddenly this isn't imported from stylesheet anymore...
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["font.size"] = 14
fs = mpl.rcParams["font.size"]


def compareReconstructedWorms(cl, eigenworms, avTrue, thetaTrue,pc3, avP, tP, tPnew):
    pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl, eigenworms)
    pc3New, pc2New, pc1New = pcsNew
    cl2 = dh.calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint)
    cl = cl[:,:-1,:]
    cl = ndimage.gaussian_filter1d(cl, 5, 1)
    # transform eigenworms exactly the same way. Otherwise we get some artefacts from nans
    r =(pcsNew[2]**2+pcsNew[1]**2)
    r = np.repeat(np.median(r), len(r))
    #lengths = 5
    #=============================================================================#
    # here we reconstruct from the true angular velocity to check the math. This is smoothed, so we need to compare with this version
    #=============================================================================#
#    plt.figure()
#    plt.plot(np.cumsum(avP))
#    plt.plot(np.cumsum(avTrue))
#    plt.plot(thetaTrue)
#    plt.show()
    xt, yt, zt = dh.recrWorm(avTrue, pc3, thetaTrue, r=r, show=0)
    pcsR = np.vstack([zt,yt, xt])
    clApprox = dh.calculateCLfromEW(pcsR, eigenworms, meanAngle, lengths, refPoint)
    #=============================================================================#    
    # reconstruct worm from predicted angular velocity.
    #=============================================================================#
    xP, yP, zP = dh.recrWorm(avP, tP, thetaTrue,r=r, show=0)
    pcsP = np.vstack([zP,yP, xP])
    clPred = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
    #=============================================================================#    
    # reconstruct worm from non-linear corrected turns
    #=============================================================================#
    xP, yP, zP = dh.recrWorm(avP, tPnew, thetaTrue,r=r)
    pcsP = np.vstack([zP,yP, xP])
    cl3 = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
    #=============================================================================#    
    # reconstruct worm from non-linear corrected turns
    #=============================================================================#
    # center around midpoint
    originalCMS = np.tile(np.mean(cl2, axis=1)[:,np.newaxis,:], (1,99,1))
    clApprox -= originalCMS
    cl -= originalCMS
    cl2 -= originalCMS
    clPred -= originalCMS
    cl3 -= originalCMS
    return cl, cl2,  clPred, cl3


def main():
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
    
    ################################################
    #
    # create figure 1: This is twice the normal size
    #
    ################################################
    # we will select a 'special' dataset here, which will have all the individual plots
    # select a special dataset - moving AML32. Should be the same as in fig 2
    movingAML32 ='BrainScanner20170613_134800'
    
    # 
    moving = data['AML32_moving']['input'][movingAML32]
    movingAnalysis = data['AML32_moving']['analysis'][movingAML32]
    
    fig = plt.figure('Fig - 4 : Recreating postures', figsize=(9.5, 9.5))
    # this gridspec makes one example plot of a heatmap with its PCA
    gs1 = gridspec.GridSpec(3, 1, height_ratios=[1,1,0.6])
    gs1.update(left=0.058, right=0.99, wspace=0.25, bottom = 0.075, top=0.95, hspace=0.25)
    # third row
    gsNL = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs1[2,:], wspace=0.4, hspace=0.5)#, width_ratios = [1,1,1,1])
    #ax3= plt.subplot(gsNL[0,0])
    #moveAxes(ax3, 'scalex', 0.05)
    #moveAxes(ax3, 'scaley', -0.1)
    #moveAxes(ax3, 'right', 0.025)
    ax1 = plt.subplot(gs1[0,0], aspect='equal')
    ax2 = plt.subplot(gs1[1,0], aspect='equal')
    moveAxes(ax1, 'left', 0.01)
    moveAxes(ax2, 'left', 0.01)
    for ax in [ax1, ax2]:
        moveAxes(ax, 'scale', 0.03)
        moveAxes(ax, 'right', 0.03)
    moveAxes(ax2, 'up', 0.025)
    moveAxes(ax1, 'up', 0.025)
        #moveAxes(ax, 'left', 0.03)
    
    ax5 = plt.subplot(gsNL[0])
    ax6 = plt.subplot(gsNL[1])
    #ax8 = plt.subplot(gsNL[2,0])
    ax9 = plt.subplot(gsNL[2])
    for ax in [ax5, ax6, ax9]:
        moveAxes(ax, 'scale', -0.02)
        #moveAxes(ax, 'scalex', -0.02)
        moveAxes(ax, 'left', -0.03)
    #for ax in [ ax9]:
    #    moveAxes(ax, 'scale', 0.02)
    #    moveAxes(ax, 'left', 0.055)
    #    moveAxes(ax, 'down', 0.02)
    #
    #moveAxes(ax6, 'down', 0.05)    
    
    #add a,b,c letters, 9 pt final size = 18pt in this case
    letters = ['A', 'B']
    x0 = 0.0
    locations = [(x0, 0.99),  (x0,0.65)]
    for letter, loc in zip(letters, locations):
        plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
                horizontalalignment='left',verticalalignment='top',)
                
    letters = ['C', 'D', 'E']
    y0 = 0.28
    locations = [(0,y0),  (0.36,y0), (0.69,y0), (0.84,0.31)]
    for letter, loc in zip(letters, locations):
        plt.figtext(loc[0], loc[1], letter, weight='bold', size=18,\
                horizontalalignment='left',verticalalignment='top',)
    # recreate worms
    #=============================================================================#
    #                           # load eigenworms
    #=============================================================================#
    eigenwormsFull = dh.loadEigenBasis(filename='../../utility/Eigenworms.dat', nComp=7, new = True)
    eigenwormsFull= dh.resize(eigenwormsFull, (7,100))
    eigenworms = dh.loadEigenBasis(filename='../../utility/Eigenworms.dat', nComp=3, new = True)    
    
    
    pc1, pc2, pc3, avTrue, thetaTrue = moving['Behavior']['Eigenworm1'],moving['Behavior']['Eigenworm2'],\
                            moving['Behavior']['Eigenworm3'],  moving['Behavior']['AngleVelocity'],  moving['Behavior']['Theta']
    #thetaTrue2 = np.unwrap(np.arctan2(pc2, pc1))
    
    #plt.plot(thetaTrue2)
    #plt.show(block=True)
    pcs = np.vstack([pc3,pc2, pc1])
    # actual centerline
    cl= moving['CL']
    #cl = dh.resize(cl, (cl.shape[0], 101, cl.shape[2]))
    #pcsNew, meanAngle, lengths, refPoint = dh.calculateEigenwormsFromCL(cl,eigenwormsFull)
       
    #cl = dh.calculateCLfromEW(pcsNew, eigenwormsFull, meanAngle, lengths, refPoint)
    #=============================================================================#
    # load predicted worm
    #=============================================================================#
    avP = movingAnalysis['ElasticNet']['AngleVelocity']['output'][:len(pcs[0])]
    tP = movingAnalysis['ElasticNet']['Eigenworm3']['output'][:len(pcs[0])]
    print 'R2'
    print  movingAnalysis['ElasticNet']['AngleVelocity']['score'], movingAnalysis['ElasticNet']['AngleVelocity']['scorepredicted']
    print  movingAnalysis['ElasticNet']['Eigenworm3']['score'], movingAnalysis['ElasticNet']['Eigenworm3']['scorepredicted']
    
    
    ################################################
    #
    #get data
    ################################################
    # weights
    flag = 'ElasticNet'
    # one example
    time = moving['Neurons']['Time']
    label = 'AngleVelocity'
    splits = movingAnalysis['Training']
    train, test = splits[label]['Train'], splits[label]['Test']
    t = moving['Neurons']['Time'][test]
    
    # calculate non-linear fits for many datasets
    #=============================================================================#    
    # correct undershoots in turns
    #=============================================================================#
    r2s = []
    dataAll = []
    for key in ['AML32_moving', 'AML70_chip']:
        dset = data[key]['input']
        res = data[key]['analysis']
        for idn in dset.keys():
            xdata1 =res[idn]['ElasticNet']['Eigenworm3']['output']
            ydata1 = dset[idn]['Behavior']['Eigenworm3']
            testL = res[idn]['Training']['Eigenworm3']['Test']
            trainL = res[idn]['Training']['Eigenworm3']['Train']
            
            #dataAll.append([xdata1[trainL], ydata1[trainL]])
            dataAll.append([xdata1, ydata1])
    
    # fit all datasets jointly
    xD, yD = np.concatenate(dataAll, axis=1)
    # non-linearity corrected
    #=============================================================================#    
    # correct undershoots in turns
    #=============================================================================#
    indices = np.argsort(tP)
    xdata, ydata = tP[indices], pc3[indices]
    
    # bin the data
    def fitfun(x,A,m,s):
        # taper off towards the ends, otherwise overshoot horribly
        #return x + A*erf(x/s) -m
        
        #return (-5)*(-0.5*(np.sign(x+5)-1)) +x
        #return 1.2*(x)#**3 +m*x +s*x**2
        #return 2*x#A*x + s*x**3 + m
        #return x +(2*erf(x/5))#*x
        #return A*(x +np.abs(erf(x/5)))
        #return A*(np.abs(erf(x/s)) +m)#*x
        return A*erf(x/s) - m
        #return x + x[np.abs(x)>10]*1.5
        #return A*(x/s-m)**3
        #return A/(1+np.exp(s*x))+m#(A*(1-np.exp(-(x-m)**2/s**2))-1)#*x
    
    p0 = [10,100,100]
    popt, pcov = curve_fit(fitfun, xD, yD, p0, sigma= 1./(np.abs((yD))))#,bounds=[[-2,-10, -10], [15,10,10]])
    
    # correct turning amplitude
    tPnew = fitfun(tP, *popt) 
   
    
    cl2, clApprox, clPred, cl3 = compareReconstructedWorms(cl, eigenworms, avTrue, thetaTrue, pc3, avP, tP, tPnew)
    
    
    #plt.show(block=True)
    mse = np.mean(np.sqrt(np.sum((clPred - cl2)**2, axis=2)), axis=1)
    mse2 = np.mean(np.sqrt(np.sum((clApprox - cl2)**2, axis=2)), axis=1)
    #plt.figure()
    #yN = fitfun(xD, *popt)
    #plt.scatter(yD, yD-xD*1.5, alpha=0.05, marker='.')
    #plt.scatter(yD, yD-xD, alpha=0.05, marker='.')
    #plt.show()
    #plt.plot(mse[test])
    #plt.plot(pc3[test])
    #plt.show()
    msePosture = []
    r2s = []
    
    dataAll = []
    for key in ['AML32_moving', 'AML70_chip']:
        dset = data[key]['input']
        res = data[key]['analysis']
        for idn in dset.keys():
            xdataL = res[idn]['ElasticNet']['Eigenworm3']['output']
            ydataL = dset[idn]['Behavior']['Eigenworm3']
            testL = res[idn]['Training']['Eigenworm3']['Test']
            trainL = res[idn]['Training']['Eigenworm3']['Train']
            indicesL = np.argsort(xdataL[trainL])
            xdataS= xdataL[indicesL]
            ydataS= ydataL[indicesL]
            # correct with fitted nonlinearit
            tNew = fitfun(xdataL, *popt)
            #use joined fit fun
            #r2s.append([r2_score( ydataL[testL],xdataL[testL]), r2_score(ydataL[testL],tNew[testL])])
            #r2s.append([r2_score( ydataL[testL],xdataL[testL]), r2_score(ydataL[testL],tNew[testL])])
            r2s.append([pearsonr( ydataL[testL],xdataL[testL])[0]**2, pearsonr(ydataL[testL],tNew[testL])[0]**2])
            # use mse - reconstruct and calculate mse for each worm
            clTmp = dset[idn]['CL']
            
            avPTmp = res[idn]['ElasticNet']['AngleVelocity']['output']
            avTmp = dset[idn]['Behavior']['AngleVelocity']
            thetaTmp = dset[idn]['Behavior']['Theta']
            
            _, clApproxTmp, clPredTmp, cl3Tmp = compareReconstructedWorms(clTmp, eigenworms, avTmp, thetaTmp,ydataL, avPTmp, xdataL,tNew)
            mseLTmp = np.mean(np.sqrt(np.sum((clPredTmp - clApproxTmp)**2, axis=2)), axis=1)
            mseNLTmp = np.mean(np.sqrt(np.sum((cl3Tmp - clApproxTmp)**2, axis=2)), axis=1)
    
            print 'MSE_L:', np.mean(mseLTmp[testL])
            print 'MSE_NL:', np.mean(mseNLTmp[testL])
            #msePosture.append([np.mean(mseLTmp[testL]), np.mean(mseNLTmp[testL])])
            turns = np.where(np.abs(ydataL[testL])>10)
            msePosture.append([np.mean(mseLTmp[testL][turns]), np.mean(mseNLTmp[testL][turns])])
            #cl3 = dh.calculateCLfromEW(pcsP, eigenworms, meanAngle, lengths, refPoint)
            #r2s.append([mean_squared_error( ydataL[testL],xdataL[testL]), mean_squared_error(ydataL[testL],fitfun(xdataL[testL], *popt))])
    
    ################################################
    #
    #first row - posture prediction
    #
    ################################################
    # find samples that are starting with a reset
    
    #loc1, loc2 = 1096, 196
    k = 1
    loc1, loc2 = 1086, 6
    #loc1, loc2 = 1096, 976
    
    offsetReset = 50
    resets = np.where((test-offsetReset)%60==0)[0]
    #print t[np.where((test-offsetReset)%60==0)[0]]
    print resets
#    plt.figure()
#    offsetx = 450
#    for li, l in enumerate(resets):
#        if np.mean(mse[test][l:l+30]) <50:
#            print l, np.mean(mse[test][l:l+30]), np.mean(mse2[test][l:l+30])
#            samplePost =test[l:l+66:6]
#            
#            for cindex, cline in enumerate(cl2[samplePost]):
#                x,y = cline[:,0], cline[:,1]
#                x -=np.mean(x)-offsetx*cindex
#                y -=np.mean(y)+li*offsetx 
#                plt.plot(x, y, 'r')
#            for cindex, cline in enumerate(clPred[samplePost]):
#                x,y = cline[:,0], cline[:,1]
#                x -=np.mean(x)-offsetx*cindex
#                y -=np.mean(y)+li*offsetx
#                plt.plot(x, y, 'k')
#            plt.text(0, li*offsetx, li)
#    plt.show()
            #ax.plot(x, y)
    # plot predicted behaviors and location of postures
    #ybeh = [10, 15]
    ##print -time[samplePost[0]]+time[samplePost[-1]]
    #for behavior, color, cpred, yl, label, align in zip(['AngleVelocity','Eigenworm3' ], \
    #            [N1, N1], [R1, B1], ybeh, ['Velocity', 'Turn'], ['center', 'center']):
    #    beh = moving['Behavior'][behavior][test]
    #    
    #    meanb, maxb = np.mean(beh),np.std(beh)
    #    beh = (beh-meanb)/maxb
    #    
    #    behPred = movingAnalysis[flag][behavior]['output'][test]
    #    behPred = (behPred-meanb)/maxb
    #    
    #    #ax3.plot(t, beh+yl, color=color)
    #    ax3.plot(t, behPred+yl, color=cpred)
    #    #ax1.text(t[test[-1]], yl, \
    #    #r'$R^2 = {:.2f}$'.format(np.float(movingAnalysis[flag][behavior]['scorepredicted'])), horizontalalignment = 'right')
    #    ax3.text(t[0]*0.6, yl, label, rotation=90, color=cpred, verticalalignment=align)
    
    
    ## add scalebar
    #l =120
    #y = ybeh[0]*0.7
    #ax3.plot([t[0], t[0]+l],[y, y], 'k', lw=2)
    #ax3.text(t[0]+l*0.5,y*0.5, '2 min', horizontalalignment='center')
    #
    #ax3.spines['left'].set_visible(False)
    #ax3.spines['bottom'].set_visible(False)
    #ax3.set_yticks([])
    #ax3.set_xticks([])
    testloc1, testloc2 =test[loc1:loc1+66:6], test[loc2:loc2+66:6]
    
    for ax, samplePost, color, timestamp in zip([ax1, ax2], [testloc2, testloc1], [N0, N0], [time[test[loc2]], time[test[loc1]]]):
        #=============================================================================#    
        # calculate mse
        #=============================================================================#
    #    plt.figure()
    #    mse = np.mean(np.sum((clPred - clApprox)**2, axis=2), axis=1)
    #    #plt.plot(mse)
    #    plt.plot(np.abs(beh)*np.max(mse)/10)
    #    print np.where(mse<1)
    #    plt.show()
        #=============================================================================#    
        # create finite-width worms
        #=============================================================================#
        # offsets to create aspect ratio and spacing in equal proportion plots
        ax.tick_params(axis='both', which='both', length=0)
    
        offsetx = 450
        offsety = 450
        # original worms
        patches1 = []
        for cindex, cline in enumerate(cl2[samplePost]):
            x,y = cline[:,0], cline[:,1]
            x -=np.mean(x)-offsetx*cindex
            y -=np.mean(y)
            vertices = createWorm(x,y)
            patches1.append(mpl.patches.Polygon(vertices, closed=True))
            #ax.plot(x, y)
        p = mpl.collections.PatchCollection(patches1, alpha=1, color=color, edgecolor='none')
        ax.add_collection(p)
        #p = mpl.collections.PatchCollection(patches1, alpha=1, color=color, edgecolor='none', offsets=(0,offsety), transOffset=ax.transData)
        #ax.add_collection(p)
        
        
    #    # 3 eigenworm approximate worms
    #    patches = []
    #    for cindex, cline in enumerate(clApprox[samplePost]):
    #        x,y = cline[:,0], cline[:,1]
    #        x -=np.mean(x)-offsetx*cindex
    #        y -=np.mean(y)+offsety
    #        vertices = createWorm(x,y)
    #        patches.append(mpl.patches.Polygon(vertices, closed=True))
    #        #ax.plot(x, y)
    #    p = mpl.collections.PatchCollection(patches, alpha=1, color=N0, edgecolor='none')
    #    ax.add_collection(p)
        
        # predicted worms
        patches = []
        for cindex, cline in enumerate(clPred[samplePost]):
            x,y = cline[:,0], cline[:,1]
            x -=np.mean(x)-offsetx*cindex
            y -=np.mean(y)+1.5*offsety
            vertices = createWorm(x,y)
            patches.append(mpl.patches.Polygon(vertices, closed=True))
            #ax.plot(x, y)
        p = mpl.collections.PatchCollection(patches, alpha=1, color=L1, edgecolor='none')
        ax.add_collection(p)
        
    #    # 3 eigenworm approximate worms
    #    patches = []
    #    for cindex, cline in enumerate(clApprox[samplePost]):
    #        x,y = cline[:,0], cline[:,1]
    #        x -=np.mean(x)-offsetx*cindex
    #        y -=np.mean(y)+offsety
    #        vertices = createWorm(x,y)
    #        patches.append(mpl.patches.Polygon(vertices, closed=True))
    #        #ax.plot(x, y)
    #    p = mpl.collections.PatchCollection(patches, alpha=1, color=N1, edgecolor='none')
    #    ax.add_collection(p)
        
        # predicted worms - with nonlinearity
        patches = []
        for cindex, cline in enumerate(cl3[samplePost]):
            x,y = cline[:,0], cline[:,1]
            x -=np.mean(x)-offsetx*cindex
            y -=np.mean(y)+2.5*offsety
            
            vertices = createWorm(x,y)
            patches.append(mpl.patches.Polygon(vertices, closed=True))
            #ax.plot(x, y)
        
        p = mpl.collections.PatchCollection(patches, alpha=1, color=L3, edgecolor='none')
        ax.add_collection(p)
        
        # add original to all lines
        for off in [ 1.5*offsety,2.5*offsety]:
            patches1 = []
            for cindex, cline in enumerate(cl2[samplePost]):
                x,y = cline[:,0], cline[:,1]
                x -=np.mean(x)-offsetx*cindex
                y -=np.mean(y) + off
                vertices = createWorm(x,y)
                patches1.append(mpl.patches.Polygon(vertices, closed=True))
                #ax.plot(x, y)
            p = mpl.collections.PatchCollection(patches1, alpha=1, color=color, edgecolor='none', zorder=-1)
            ax.add_collection(p)
        
        # add horizontal line and text
        #ax.axhline(-2*offsetx,color = 'k', linestyle =':')
        ax.text(0.5, 0.65,'predicted from neural activity', transform = ax.transAxes, horizontalalignment = 'center')
        ax.text(0.5, 0.95,'measured postures (t = {} s)'.format(int(timestamp)), transform = ax.transAxes, horizontalalignment = 'center')
        # general plot style
        ax.set_xticks(np.arange(0,len(samplePost+1)*offsetx, 2*offsetx))
        ax.set_xticklabels(['{} s'.format(i) for i in range(0, len(samplePost)+1, 2)])
        ax.set_xlim([-300, 4700])
        ax.set_ylim([-2.5*offsety-350, 350])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    # just left plot
    
    ax1.set_yticks([0,-1.5*offsety, -2.5*offsety])
    ax1.set_yticklabels(['original',  'SLM', 'SLM+NL'])
    
    ax2.set_yticks([0, -1.5*offsety, -2.5*offsety])
    ax2.set_yticklabels(['original',  'SLM', 'SLM+NL'])
    # move to right
    #moveAxes(ax1, 'right', -0.07)
    #moveAxes(ax2, 'right', -0.07)
    # remove xticks from first axes
    cleanAxes(ax1, 'x')
    ax2.spines['bottom'].set_visible(True)
    
    ################################################
    #
    #third row - non-linearity
    #
    ################################################
    #ax4 = plt.subplot(gsNL[0,1])
    # show non-linearity for one neuron
    #ax4.scatter(xdata, ydata, color='k', alpha=0.01)
    
    #ax4.plot(xdata, tPnew[indices], color=R1, linestyle='--', label='fit')
    #ax4.set_xlabel('Predicted turn')
    #ax4.set_ylabel('True turn', labelpad=-10)
    #moveAxes(ax4, 'right', 0.02)
    
    
    
    ax6.axvspan(time[test[loc1]], time[test[loc1+60]], color=N2, zorder=-10, alpha=1, ymax=0.75, edgecolor='None')
    ax6.axvspan(time[test[loc2]], time[test[loc2+60]], color='k', zorder=-10, alpha=1, ymax=0.75, fill=False, edgecolor='k', hatch='\\\\\\\\')
    # compare to linear
    #ax6.plot(time[test], pc3[test], color=N1, label='true', zorder=-1, linestyle=':')
    ax6.plot(time[test], tP[test], color=L1, label='linear',linestyle='--', zorder=20)
    #ax7.plot(time[test], tP[test], color=R1, label='linear', zorder=10)
    # compare to nonlinear
    ax6.plot(time[test], pc3[test], color=N0, label='true', zorder=-1, linestyle=':')
    ax6.plot(time[test], tPnew[test], color=L3,linestyle='-', label='non-linear')
    
    r2orig = r2_score(pc3[test], tP[test])
    r2nl = r2_score(pc3[test], tPnew[test])
    #r2orig = pearsonr(pc3[test], tP[test])[0]**2
    #r2nl = pearsonr(pc3[test], tPnew[test])[0]**2
    ax6.text(1,1.1,r'$r_L^2 = {:.2f}$'.format(r2orig), horizontalalignment = 'right', transform = ax6.transAxes)
    #ax6.text(-0.1,0.5,r'Turn', horizontalalignment = 'center', transform = ax6.transAxes, rotation =90)
    ax6.text(1,1,r'$r_{{NL}}^2 = {:.2f}$'.format(r2nl), horizontalalignment = 'right',verticalalignment = 'top', transform = ax6.transAxes)
    #ax6.legend(loc=8, bbox_to_anchor=(-0.1, 0.5))
    #ax7.spines['left'].set_visible(False)
    #ax6.spines['left'].set_visible(False)
    #ax7.set_yticks([])
    #ax6.set_yticks([])
    #ax6.plot([time[test][0], time[test][0]], [-15,-5], color=B1, lw=2)
    #ax6.text(time[test][0]*0.99, -10,'10', color=B1, verticalalignment='center', horizontalalignment='right')
    ax6.set_ylabel('Body curvature')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylim([-20, 20])
    #ax7.set_xlabel('Time (s)')
    # move
    #moveAxes(ax6, 'right', 0.015)
    #moveAxes(ax7, 'left', 0.025)
    
    fitx = np.arange(-15,15, 0.1)
    ax5.plot(fitx, fitfun(fitx, *popt), color='r', alpha=0.75)
    ax5.scatter(xD, yD, s=3, c='k', alpha=0.01)
    ax5.set_xlabel('Predicted \n body curvature')
    ax5.set_ylabel('Actual \n body curvature')
    
    
    #mkStyledBoxplot(ax8, [0,1], np.reshape(fitdata, (1,-1)),[L1], ['A'])
    #mkStyledBoxplot(ax8, [0,1], np.array(r2s).T,[L1, L3], ['L', 'NL'])
    #print 'R2_scores', np.mean(r2s, axis=0)
    #ax8.set_xlim([-0.5,1.5])
    #ax8.set_ylabel(r'$r^2$')
    
    
    mkStyledBoxplot(ax9, [0,1], np.array(msePosture).T,[L1, L3], ['SLM', 'SLM+NL'], rotate=False)
    print 'MSE_scores', np.mean(msePosture, axis=0)
    ax9.set_xlim([-0.5,1.5])
    ax9.set_ylabel(r'RMSE (BC > 10)')
    plt.show()

main()