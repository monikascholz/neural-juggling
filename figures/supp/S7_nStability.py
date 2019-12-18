#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:56:27 2018
Show the variablity of the sparseness selection.
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

from sklearn.linear_model import ElasticNet, ElasticNetCV

from scipy.interpolate import interp1d
from scipy.optimize import newton 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
np.random.seed(13)

def findBounds(x, y, ybound):
    """move by one stdeviation to increase regularization."""
    #yUp = np.min(y) + std[np.argmin(y)]
    yFunc =  interp1d(x,y,'cubic',fill_value='extrapolate')
    xalpha = x[np.argmax(y)]
    bounds = []
    for xalpha in [1.5*xalpha, 0.9*xalpha]:
        # start high for upper bound
        try:
            xUpper = newton(lambda x: yFunc(x) - ybound, xalpha, tol=1.48e-06)
        except RuntimeError:
            xUpper = xalpha
        bounds.append(xUpper)
#            xalpha = x[np.argmax(y)]*0.9
#            xUpper = newton(lambda x: yFunc(x) - ybound, xalpha, tol=1.48e-06)
#            plt.figure()
#            plt.plot(x,y, 'ro')
#            plt.plot(x,yFunc(x), 'r-')
#            plt.axhline(ybound)
#            plt.show()
#            plt.figure()
#            plt.plot(x, yFunc(x) - ybound)
#            plt.show()
        # start low for lower bound
        #xalpha = x[np.argmin(y)]*0.25
        #xLower = newton(lambda x: np.abs(yFunc(x) - ybound), xalpha)
        #print xUpper, xalpha, xLower
        
        #plt.axvline(xLower)
    #    plt.axvline(xUpper)
    #    
    #    plt.show()
    return bounds

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

fig = plt.figure('S7_Sparseness', figsize=(9.5,9.5))
# this is a gridspec
gs1 = gridspec.GridSpec(3, 2)
gs1.update(left=0.09, right=0.99,  bottom = 0.07, top=0.95, hspace=0.25, wspace=0.25)

## add a,b,c letters, 9 pt final size = 18pt in this case
letters = ['A', 'B', 'C', 'D', 'E', 'F']
y0 = 0.96
y1 = 0.66
y2 = 0.33
locations = [(0,y0),  (0.5,y0), (0.,y1), (0.5,y1), (0, y2), (0.5, y2)]
for letter, loc in zip(letters, locations):
    plt.figtext(loc[0], loc[1], letter, weight='semibold', size=18,\
            horizontalalignment='left',verticalalignment='baseline',)


ax2 = plt.subplot(gs1[2,0])
ax3 = plt.subplot(gs1[2,1])

####################
# get lagged r2s
#########################
a = 0.4
behaviors = ['AngleVelocity', 'Eigenworm3']
flag = 'ElasticNet'
colors = [R1, B1]
alphaFit = [np.logspace(-4,-1,20), np.logspace(-2.5,-0.5,20)]

for i in range(2):
    scores = []
    weights = []
    alphas = []
    alphaStar = []
    NRange = []
    Ntotal = []
    ax = plt.subplot(gs1[i,0])
    ax.set_ylabel(r'R$^2$')
    ax.set_ylim([-1, 1])
    ax1 = plt.subplot(gs1[i,1])
    ax1.set_ylabel('# Neurons')
    if i==1:
        ax.set_xlabel(r'$\alpha$')
        ax1.set_xlabel(r'$\alpha$')
    
    for key in ['AML32_moving', 'AML70_chip']:
        for k in data[key]['analysis'].keys():
            w = []
            s = []
            
            dset = data[key]['analysis'][k]
            l1 = dset[flag][behaviors[i]]['l1_ratio']
            alpha = dset[flag][behaviors[i]]['alpha']
            training = data[key]['analysis'][k]['Training'][behaviors[i]]
            train, test = training['Train'], training['Test']
            X = np.copy(data[key]['input'][k]['Neurons']['Activity']).T
            y = np.copy(data[key]['input'][k]['Behavior'][behaviors[i]])
            scale = True
            if scale:
                scalerX = preprocessing.StandardScaler().fit(X[train])
                yt = np.reshape(y, (-1,1))
                scalerY = preprocessing.StandardScaler().fit(yt[train])  
                #scale data
                X = scalerX.transform(X)
                y = scalerY.transform(yt)
            
            alph = alphaFit[i]
            alphaStar.append(alpha)
            Ntotal.append(1.0*len(dset[flag][behaviors[i]]['weights']))
            
#            fold = TimeSeriesSplit(n_splits=5, max_train_size=None)
#            reg = ElasticNetCV(l1, cv=fold, verbose=0, selection='random', alphas=alph, tol = 1e-10)      
#            reg.fit(X[train], y[train])
#            scorepred = reg.score(X[test], y[test])
#            score = reg.score(X[train], y[train])
#            print scorepred, len(reg.coef_[np.abs(reg.coef_)>0])
            weightDistro = []
            for alpha in alph:
                reg = ElasticNet(alpha=alpha, l1_ratio = l1)                
                #reg = ElasticNetCV(l1, cv=fold, verbose=0, selection='random', alphas=[alpha])      
                reg.fit(X[train], y[train])
                scorepred = reg.score(X[test], y[test])
                score = reg.score(X[train], y[train])
                #XXX TODO should just get the alpha and mse path for each recording, cal 1 stdev and then
                # calculate the Ns
                s.append(scorepred)
                w.append(len(reg.coef_[np.abs(reg.coef_)>0]))
                
                weightDistro.append(reg.coef_)
            # show weight dirctibution as a function of alpha
#            plt.figure()
#            plt.imshow(np.array(weightDistro))
#            plt.show()
#            weightDistro
            weights.append(w)
            scores.append(s)
            alphas.append(alph)
            
    scores = np.array(scores)
    weights = np.array(weights)
    alphas = np.array(alphas)
    # scan across alphas
    ax.plot(alphas.T, scores.T, color=colors[i])
    ax1.plot(alphas.T, weights.T, color = colors[i])
    
    
    # calculate confidence boundaries
    for j in range(len(scores)):
        # find which sampled alpha is closest to alphaStar
        alphaIdx = np.argmin(np.abs(alphas[j] - alphaStar[j]))
        N = weights[j][alphaIdx]
        # find the corresponding R2 score
        score = scores[j][alphaIdx] 
        # find the bounds
        bounds = findBounds(alphas[j], scores[j], ybound = score*0.95)
        ntmp = [N]
        for b in bounds:
            aBoundIdx = np.argmin(np.abs(alphas[j] - b))
            # find the corresponding n
            Nb = weights[j][aBoundIdx]
            ntmp.append(Nb)
            print ntmp
        ntmp.append(Ntotal[j])
        NRange.append(ntmp)
    print NRange
    NRange = np.array(NRange)
    print NRange.shape
    # plot a representation of the bounds
    for k, ns in enumerate(NRange):
        ax2.plot([k*0.1 +i, k*0.1+i],[np.min(ns[0:3]), np.max(ns[0:3])], colors[i])
        ax2.plot([k*0.1+i],ns[0], colors[i], marker='o')
    ax2.set_ylabel('# Neurons')
    ax2.set_xticks([0+0.1*len(NRange)/2., 1+0.1*len(NRange)/2.])
    ax2.set_xticklabels(['Velocity', 'Body \n curvature'])
    # plot a representation of the bounds - in percent
    for k, ns in enumerate(NRange):
        ax3.plot([k*0.1 +i, k*0.1+i],[np.min(ns[0:3]), np.max(ns[0:3])]/ns[3], colors[i])
        ax3.plot([k*0.1+i],ns[0]/ns[3], colors[i], marker='o')
    ax3.set_ylabel('Fraction \n of Neurons')
    ax3.set_xticks([0+0.1*len(NRange)/2., 1+0.1*len(NRange)/2.])
    ax3.set_xticklabels(['Velocity', 'Body \n curvature'])
    
    #for k, ns in enumerate(NRange):
       # a#x3.plot([k*0.1 +i, k*0.1+i],ns/1.0/ns[0]*, colors[i])
#    mkStyledBoxplot(ax3, [i*0.25, i*0.25+0.5], y_data= [100*np.squeeze((np.abs(np.diff(NRange))).T/1.0/NRange[:,0])], \
#                    clrs=[colors[i]], lbls=['Velocity'], scatter = True, rotate=0, dx=None)
#    ax3.set_ylabel('% Change')
#    ax3.set_xlim(-0.15, 0.35)
#    ax3.set_xticks([0, 0.25])
#    ax3.set_xticklabels(['Velocity', 'Body \n curvature'])
                
plt.show()

