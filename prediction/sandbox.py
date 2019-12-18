# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 2017
sandboxing

@author: mscholz
"""

import os
from prediction import userTracker
path = userTracker.dataPath()
print path
folder = os.path.join(path, '{}_{}/'.format(typ, condition))
dataLog = os.path.join(path,'{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
outLocData = os.path.join(path,'/Analysis/{}_{}.hdf5'.format(typ, condition))
print folder


import numpy as np
#import fourier_vec as fourier
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io
import os
import scipy.linalg as linalg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
#import pyemma.coordinates as coor
import pycpd as cpd
# standard modules
import numpy as np
import matplotlib.pylab as plt
import h5py
# custom modules 
import dataHandler as dh
import makePlots as mp
import dimReduction as dr
from sklearn.decomposition import PCA

from pycpd import deformable_registration, rigid_registration


def runReg(X,Y,Y0, dim3, registration, **kwargs):
    """registatrion: pycpd function"""
    from functools import partial
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    import numpy as np
    import time
    
     
    def visualize(iteration, error, X, Y, ax):
        if dim3:
            plt.cla()
            ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', s=1)
            ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', s=1)
            plt.draw()
            print("iteration %d, error %.9f" % (iteration, error))
            plt.pause(0.1)
        else:
            plt.cla()
            ax.scatter(X[:,0],  X[:,1], color='red', s=5)
            ax.scatter(Y[:,0],  Y[:,1], color='blue', alpha=0.5, s=5)
            plt.draw()
            print("iteration %d, error %.9f" % (iteration, error))
            plt.pause(0.1)
    
    if dim3:
        fig = plt.figure()
        ax1 = fig.add_subplot(211, projection='3d')
        ax1.scatter(X[:,0],  X[:,1], X[:,2], color='red')
        ax1.scatter(Y0[:,0],  Y0[:,1], Y0[:,2], color='blue')
        ax = fig.add_subplot(212, projection='3d')
        ax.scatter(X[:,0],  X[:,1], X[:,2], color='red')
        ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue')
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.scatter(X[:,0],  X[:,1], color='red')
        ax1.scatter(Y0[:,0],  Y0[:,1], color='blue')
        ax = fig.add_subplot(212)
        ax.scatter(X[:,0],  X[:,1], color='red')
        ax.scatter(Y[:,0],  Y[:,1], color='blue')
    #reg = cpd.deformable_registration(X, Y, tolerance=1e-6)
    callback = partial(visualize, ax=ax)
    reg = registration(X, Y, **kwargs)
    reg.register(callback)
    
    plt.show()
    # return deformed Y
    return reg

def loadAtlas(dim3=False):
    # use 3d neuron atlas positions?
    if dim3:
        neuronAtlasFile = 'utility/NeuronPositions.mat'
        neuronAtlas = scipy.io.loadmat(neuronAtlasFile)
        # load matlab neuron positions from atlas
        Xref = np.hstack([neuronAtlas['x'], neuronAtlas['y'],neuronAtlas['z']])
        
        relevantIds = (Xref[:,1]<-2.5)#*(Xref[:,1]<-3.0)
        Xref = Xref[relevantIds]
        
        labels = np.array([str(idn[0][:3]) for idn in neuronAtlas['ID'][relevantIds, 0]])
        #Xref = Xref[Xref[:,0]<2.7]
    # load 2d new location file
    else:
        neuron2D = 'utility/celegans277positionsKaiser.csv'
        labels = np.loadtxt(neuron2D, delimiter=',', usecols=(0), dtype=str)
        neuronAtlas2D = np.loadtxt(neuron2D, delimiter=',', usecols=(1,2))
        relevantIds = (neuronAtlas2D[:,0]>-0.0)#*(Xref[:,0]<0.1)
        Xref = neuronAtlas2D[relevantIds]
        Xref[:,0] = -Xref[:,0]
        labels = labels[relevantIds]
    return Xref, labels
###############################################    
# 
#    run parameters
#
###############################################
typ = 'AML32' # possible values AML32, AML18, AML70
condition = 'moving' # Moving, immobilized, chip
first = True # if true, create new HDF5 file
###############################################    
# 
#    load data into dictionary
#
##############################################
folder = '{}_{}/'.format(typ, condition)
dataLog = '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
outLoc = "Analysis/{}_{}_results.hdf5".format(typ, condition)
outLocData = "Analysis/{}_{}.hdf5".format(typ, condition)
# data parameters
dataPars = {'medianWindow':1, # smooth eigenworms with gauss filter of that size, must be odd
            'gaussWindow':15, # sgauss window for angle velocity derivative. must be odd
            'rotate':False, # rotate Eigenworms using previously calculated rotation matrix
            'windowGCamp': 5,  # gauss window for red and green channel
            'interpolateNans': 5,#interpolate gaps smaller than this of nan values in calcium data
            }

dataSets = dh.loadDictFromHDF(outLocData)
keys = dataSets.keys()
results = dh.loadDictFromHDF(outLoc) 
dim3 = True
Xref, labels = loadAtlas(dim3=dim3)

Xref -= np.mean(Xref, axis=0)
Xref /=np.max(Xref, axis=0)
Y = dataSets[keys[1]]['Neurons']['Positions'].T

Y -= np.mean(Y, axis=0)
Y /=np.max(Y, axis=0)
if not dim3:
    Y = Y[:,:2]
    
pca = PCA(n_components = 2)
Xref = pca.fit_transform(Xref)    
Y = pca.fit_transform(Y)
dim3 = False

print Xref.shape, Y.shape
registration = cpd.rigid_registration
reg = runReg(Xref,Y,Y, dim3, registration, tolerance=1, maxIterations=150)
registration = cpd.deformable_registration
reg = runReg(Xref,reg.TY,reg.TY, dim3, registration, tolerance=1e-6, maxIterations=50)
order = np.argsort(np.argmax(reg.P, axis = 1))
print order.shape
plt.subplot(211)
plt.imshow(reg.P[order], aspect='auto')
ax1=plt.subplot(212)
#ax1.scatter(Xref[:,0],  Xref[:,1], color='red')
ax1.scatter(reg.TY[:,0],  reg.TY[:,1], color='blue')
for loc, idn in  zip(Xref,labels):
    ax1.text(loc[0], loc[1], idn, verticalalignment='center', horizontalalignment='center')
plt.show()

results = dh.loadDictFromHDF(outLoc)
idents = []
control = []
shuffled_labels = labels.copy()
np.random.shuffle(shuffled_labels)
for weights in [results[keys[1]]['LASSO']['AngleVelocity'], results[keys[1]]['LASSO']['Eigenworm3']]:
    print weights['scorepredicted']    
    for windex, w in enumerate(weights['weights']):
        if w !=0:
            print w, labels[np.argsort(reg.P[:,windex])][-3:]
            idents.append(labels[np.argsort(reg.P[:,windex])][-3:])
            control.append(shuffled_labels[np.argsort(reg.P[:,windex])][-3:])
            
            
av_neurons = ['ASI', 'AIY','AIB', 'RIM', 'SMB', 'RMD', 'SMD']
turn_neurons = ['RIV', 'SMD']
motionneurons = ['ASI', 'AIY','AIB', 'RIM', 'SMB', 'RMD', 'SMD', 'RIV']
counter = [0, 0]
for neuron in np.concatenate(idents):
    if neuron in motionneurons:
        counter[0] += 1

for neuron in np.concatenate(control):
    if neuron in motionneurons:
        counter[1] += 1
    
print 'Fraction motion neurons in exp:', counter[0]/1.0/len(np.concatenate(idents))
print 'Fraction motion neurons in control:', counter[1]/1.0/len(np.concatenate(control))
   
 
 
#binx, biny = 20,20
#
#wormPos = []
## for now hard code ventral
#ventral = [-1,1,1]
#for lindex, line in enumerate(np.loadtxt(dataLog, dtype=str, ndmin = 2)[:1]):
#    folderName = folder.format(line[0])
#
#    pts = np.array(dh.loadPoints(folderName, straight = True))
#    nNeur = [len(pt) for pt in pts]
#    Y0 = np.array(pts[np.max(nNeur)])# + 10*np.ones(X.shape)+5*(0.5-np.random.random_sample(X.shape))#10*np.ones(X.shape)#
#    pts = np.array(dh.loadPoints(folderName, straight = False))    
#    #YS = np.array(pts[np.max(nNeur)])# + 10*np.ones(X.shape)+5*(0.5-np.random.random_sample(X.shape))#10*np.ones(X.shape)#
#    print 'before', Y0.shape
#    Y0 = Y0[np.isfinite(Y0[:,0])]
#    print 'after',Y0.shape
#    fig = plt.figure('Atlas and points')
#    ax = fig.add_subplot(2,1,lindex+2)
#    # invert y-axis if ventral side up
#    Y0[:,1] = Y0[:,1]*ventral[lindex]-(-1+ventral[lindex])*50
#    Y0[:,0] -= 200# Y0[:,1]*ventral[lindex]-(-1+ventral[lindex])*50
#    # make 2D
#    Y0 = Y0[:,:2]
#    ax.scatter(Y0[:,0],  Y0[:,1], color='green', alpha=0.5, s =5)
#   
#
#    
#    #ax.scatter(YS[:,0],  YS[:,1], color='k', alpha=0.5, s=1)
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    
#    
#     # plot marginals
##    H, xedge, yedge = np.histogram2d(Y0[:,0],  Y0[:,1], bins=(binx,biny))
##    fig = plt.figure('Hist')
##    ax = fig.add_subplot(10,2,2*lindex+3)    
##    plt.step(np.arange(binx), np.sum(H, axis=1))
##    ax = fig.add_subplot(10,2,2*lindex+4)    
##    plt.step(np.arange(biny), np.sum(H, axis=0))
#    #ax.set_aspect(aspect=(np.ptp(Y0[:,0])/np.ptp(Y0[:,1])))
#    wormPos.append(Y0)
#    
#fig = plt.figure('Atlas and points')
#ax = fig.add_subplot(211)
#
#ax.scatter(X[:,0],  X[:,1], color='red', s=5)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#
##H, xedge, yedge = np.histogram2d(X[:,0],  X[:,1], bins=(binx,biny))
##fig = plt.figure('Hist')
##ax = fig.add_subplot(10,2,1)  
##plt.step(np.arange(binx), np.sum(H, axis=1))
##ax = fig.add_subplot(10,2,2)  
##plt.step(np.arange(biny), np.sum(H, axis=0))
###ax.set_aspect(aspect=(np.ptp(Y0[:,0])/np.ptp(Y0[:,1])))
#  
#    
#plt.show()
##
##registration = cpd.rigid_registration
##Y0 = runReg(wormPos[0],X,X, dim3, registration, tolerance=1e-3, maxIterations=150)
##    
##    
##registration = cpd.deformable_registration
##
##
##Y0 = runReg(wormPos[0],X,X, dim3, registration, tolerance=1e-5, maxIterations=150)
##
##print X.shape
### to atlas
##rescale atlas
#Y = wormPos[0]
#X *= np.ptp(Y)/np.ptp(X)
#X = X - np.mean(X, axis=0)
#registration = cpd.rigid_registration
#Y0 = runReg(X,Y,Y, dim3, registration, tolerance=1e-3, maxIterations=150)
#    
#    
#registration = cpd.deformable_registration
#Y0 = runReg(X,Y0,Y0, dim3, registration, tolerance=1e-5, maxIterations=150)
#    
#    
#    
#    #
#for lindex, line in enumerate(np.loadtxt(dataLog, dtype=str, ndmin = 2)[1:2]):
#    folder = folder.format(line[0])
#
#    pts = np.array(dh.loadPoints(folder,straight = True))
#    print len(pts)
#    print pts.shape
#    
#    
#    
#    #rigid
#    nNeur = [len(pt) for pt in pts]
#    Y0 = np.array(pts[np.max(nNeur)])
#    fig = plt.figure('Atlas and points')
#    ax = fig.add_subplot(211)
#    
#    ax.scatter(Y0[:,0],  Y0[:,1], color='green')
#    
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_aspect(aspect=(np.ptp(Y0[:,0])/np.ptp(Y0[:,0])))
#    
#    ax = fig.add_subplot(212)
#    
#    ax.scatter(X[:,0],  X[:,1], color='red')
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_aspect(aspect=(np.ptp(Y0[:,0])/np.ptp(Y0[:,0])))
#    plt.show()
#    if not dim3:
#        Y0 = Y0[:,:2]
#        # flip x axis
#        Y0[:,1] = -Y0[:,1]
#        Y0[:,0] = -Y0[:,0]
#    print Y0.shape
#    
#    # subsample X
#    #X = X[np.random.randint(0,len(Xref), (len(Xref)-50))]
#    #SHIFT TO CENTER
#    Y0 = Y0-np.mean(Y0, axis=0)
#    Y0 += np.mean(X, axis=0)
#    #adjust range
#    Y0 /= np.ptp(Y0, axis=0)
#    Y0 *= np.ptp(X, axis=0)
#    
#    #Y0[:,0] = pos[1][:,1]
#    
#    #Y0[:,1] = np.max(Y0[:,1])-Y0[:,1]
#    Y = np.copy(Y0)
#    registration = cpd.rigid_registration
#    Y0 = runReg(X,Y,Y0, dim3, registration, tolerance=1e-3, maxIterations=100)
#    
#    
#    Y = np.copy(Y0)
#    registration = cpd.deformable_registration
#    runReg(X,Y,Y0, dim3, registration, tolerance=1e-18)
#    
#    

#threed = 1
#
#if threed:
#    from functools import partial
#    from scipy.io import loadmat
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D
#    from pycpd import deformable_registration
#    import numpy as np
#    import time
#    
#    def visualize(iteration, error, X, Y, ax):
#        plt.cla()
#        ax.scatter(X[:,0],  X[:,1], X[:,2], color='red')
#        ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue')
#        plt.draw()
#        print("iteration %d, error %.5f" % (iteration, error))
#        plt.pause(0.1)
#    
#    
#    fig = plt.figure()
#    ax1 = fig.add_subplot(211, projection='3d')
#    ax1.scatter(X[:,0],  X[:,1], X[:,2], color='red')
#    ax1.scatter(Y0[:,0],  Y0[:,1], Y0[:,2], color='blue')
#    ax = fig.add_subplot(212, projection='3d')
#    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red')
#    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue')
#    
#    
#    #reg = cpd.deformable_registration(X, Y, tolerance=1e-6)
#    callback = partial(visualize, ax=ax)
#    reg = cpd.rigid_registration(X, Y)
#    reg.register(callback)
#    Y0 = reg.TY
#    plt.show()
#    
#    fig = plt.figure('Rigid results')
#    ax = fig.add_subplot(211, projection='3d')
#    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red')
#    ax.scatter(Y0[:,0],  Y0[:,1], Y0[:,2], color='blue')
#    ax = fig.add_subplot(212, projection='3d')
#    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red')
#    
#    ax.scatter(Y0[:,0],  Y0[:,1], Y0[:,2], color='blue')
#    plt.show()
#    
#    
#    fig = plt.figure('Deformable iteration')
#    ax = fig.add_subplot(111, projection='3d')
#    callback = partial(visualize, ax=ax)
#    
#    #reg = cpd.deformable_registration(X, Y, tolerance=1e-6)
#    reg = cpd.deformable_registration(X, Y0, tolerance=1e-8, )
#    out = reg.register(callback)
#    plt.show()
#    Y0 = reg.TY
#    fig = plt.figure('Deformable Final results')
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red')
#    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue')
#    ax.scatter(Y0[:,0],  Y0[:,1], Y0[:,2], color='green')
#    plt.show()
#def rankTransform(neuroMap):
#    """takes a matrix and transforms values into rank within the colum. ie. neural dynamics: for each neuron
#    calculate its rank at the current time."""
#    temp = neuroMap.argsort(axis=0)
#    rank = temp.argsort(axis=0)
#        
#    return rank
#    
#def multicolor(ax,x,y,z,t,c, threedim = True):
#    """multicolor plot from francesco."""
#    if threedim:
#        points = np.array([x,y,z]).transpose().reshape(-1,1,3)
#        segs = np.concatenate([points[:-1],points[1:]],axis=1)
#        lc = Line3DCollection(segs, cmap=c)
#        lc.set_array(t)
#        ax.add_collection3d(lc)
#        ax.set_xlim(np.min(x),np.max(x))
#        ax.set_ylim(np.min(y),np.max(y))
#        ax.set_zlim(np.min(z),np.max(z))
#    else:
#        points = np.array([x,y]).transpose().reshape(-1,1,2)
#        segs = np.concatenate([points[:-1],points[1:]],axis=1)
#        lc = LineCollection(segs, cmap=c)
#        lc.set_array(t)
#        ax.add_collection(lc)
#        ax.set_xlim(np.min(x),np.max(x))
#        ax.set_ylim(np.min(y),np.max(y))
#        
#
#def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
#    The Savitzky-Golay filter removes high frequency noise from data.
#    It has the advantage of preserving the original shape and
#    features of the signal better than other types of filtering
#    approaches, such as moving averages techniques.
#    Parameters
#    ----------
#    y : array_like, shape (N,)
#        the values of the time history of the signal.
#    window_size : int
#        the length of the window. Must be an odd integer number.
#    order : int
#        the order of the polynomial used in the filtering.
#        Must be less then `window_size` - 1.
#    deriv: int
#        the order of the derivative to compute (default = 0 means only smoothing)
#    Returns
#    -------
#    ys : ndarray, shape (N)
#        the smoothed signal (or it's n-th derivative).
#    Notes
#    -----
#    The Savitzky-Golay is a type of low-pass filter, particularly
#    suited for smoothing noisy data. The main idea behind this
#    approach is to make for each point a least-square fit with a
#    polynomial of high order over a odd-sized window centered at
#    the point.
#    Examples
#    --------
#    t = np.linspace(-4, 4, 500)
#    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
#    ysg = savitzky_golay(y, window_size=31, order=4)
#    import matplotlib.pyplot as plt
#    plt.plot(t, y, label='Noisy signal')
#    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
#    plt.plot(t, ysg, 'r', label='Filtered signal')
#    plt.legend()
#    plt.show()
#    References
#    ----------
#    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
#       Data by Simplified Least Squares Procedures. Analytical
#       Chemistry, 1964, 36 (8), pp 1627-1639.
#    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
#       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
#       Cambridge University Press ISBN-13: 9780521880688
#    """
#    import numpy as np
#    from math import factorial
#
#    try:
#        window_size = np.abs(np.int(window_size))
#        order = np.abs(np.int(order))
#    except ValueError, msg:
#        raise ValueError("window_size and order have to be of type int")
#    if window_size % 2 != 1 or window_size < 1:
#        raise TypeError("window_size size must be a positive odd number")
#    if window_size < order + 2:
#        raise TypeError("window_size is too small for the polynomials order")
#    order_range = range(order+1)
#    half_window = (window_size -1) // 2
#    # precompute coefficients
#    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#    # pad the signal at the extremes with
#    # values taken from the signal itself
#    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
#    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
#    y = np.concatenate((firstvals, y, lastvals))
#    return np.convolve( m[::-1], y, mode='valid')
#
#
#
#def qualityCheckBehavior(T, etho, xPos, yPos, vel, pc12, pc3):
#    plt.figure()
#    ax1 = plt.subplot(321)
#    plotEthogram(ax1, T, etho)
#    plt.ylabel('Ethogram')
#    plt.subplot(322)
#    plt.plot(xPos, yPos)
#    plt.subplot(323)
#    cax1 = plt.imshow(Y, aspect='auto', interpolation='none', extent=[0,T[-1],lenY,0],vmax=1)
#    cbar = plt.colorbar(cax1)
#    plt.subplot(324)
#    plt.plot(vel)
#    plt.subplot(325)
#    plt.plot(pc12[:,0], pc12[:,1], '-')
#    plt.subplot(326)
#    plt.plot(pc3)
#    plt.tight_layout()
#
#def plotEthogram(ax, T, etho, alpha = 0.5, yVal=1):
#    """make a block graph ethogram for elegans behavior"""
#    colDict = {-1:'red',0:'k',1:'green',2:'blue'}
#    labelDict = {-1:'Reverse',0:'Pause',1:'Forward',2:'Turn'}
#    #y1 = np.where(etho==key,1,0)
#    for key in colDict.keys():
#        plt.fill_between(T, y1=np.ones(len(T))*yVal, where=(etho==key)[:,0], \
#        interpolate=False, color=colDict[key], label=labelDict[key], alpha = alpha)
#    plt.xlim([min(T), max(T)])
#    #plt.ylim([0, 1.1*yVal])
#    
#    ax.yaxis.set_visible(False)
#    plt.legend(ncol=2)
#
#######################################
##
## load and prepare data
##
######################################
##folder = '../../PanNeuronal/20171017/BrainScanner20171017_184114/' #also good one
#folder = '../../PanNeuronal/GoldStandardDatasets/BrainScanner20170610_105634_linkcopy/' #also good one
#
#data = scipy.io.loadmat(folder+'heatData.mat')
#print data.keys()
#
#Y = np.array(data['Ratio2'])
## ordering from correlation map
#order = np.array(data['cgIdx']).T[0]-1
## crop first 100 seconds due to issues in starting recording
#TZero=1200 #in frames
#Y = Y[:,TZero:]
## unpack behavior variable
#bData = data['behavior'][0][0].T
#etho, xPos, yPos, vel, pc12, pc3 = bData
#pc3 = pc3[:,0]
#etho = etho[TZero:]
#xPos = xPos[TZero:]
#yPos = yPos[TZero:]
#vel = vel[TZero:]
#pc12 = pc12[TZero:]
#pc3 = pc3[TZero:]
#
## prep neural data
#Y = Y[order]
#mask = np.isnan(Y)
#Y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Y[~mask])
#lenY = len(Y)
#T = np.arange(Y.shape[1])/6.
## in behavior there are 6 entries: ethogram, xpos,ypos, v, pc12, pc3
## -1 for reverse, 0 pause, 1 forward, 2 turn
#eigenwormplot = 0
#
#Theta = np.unwrap(np.arctan2(pc12[:,1], pc12[:,0]))
#velo = savitzky_golay(Theta, window_size=9, order=3, deriv=1, rate=1)
#turns = savitzky_golay(pc3, window_size=9, order=3, deriv=1, rate=1)
#
#neuroSmooth = np.array([savitzky_golay(line, window_size=5, order=3, deriv=0, rate=1) for line in Y])
#
#rankM = rankTransform(neuroSmooth)
#plt.subplot(211)
#cax1 = plt.imshow(rankM, aspect='auto')
#plt.colorbar()
#plt.subplot(212)
#cax1 = plt.imshow(neuroSmooth, aspect='auto', vmax =0.5, vmin=0.1)
#plt.colorbar()
#plt.show()
#neuroSmooth = rankM
#######################################
##
## Working with tica
##
######################################
#ticaObj = tica_obj = coor.tica(neuroSmooth.T, lag=2, dim =5, kinetic_map=False)
#
## plot
##plt.figure(1223)
#Y = ticaObj.get_output()[0]+2
##ax1 = plt.subplot2grid((2,2),(0,0))
##plt.plot(T,Y[:,0], label='0')
##plotEthogram(ax1, T, etho, alpha=0.5, yVal = 5)
##ax2 = plt.subplot2grid((2,2),(0,1))
##plt.plot(T,Y[:,1], label='1')
##plotEthogram(ax2, T, etho, alpha=0.5, yVal = 5)
##ax3 = plt.subplot2grid((2,2),(1,0))
##plt.plot(T,Y[:,2], label='2')
##plotEthogram(ax3, T, etho, alpha=0.5, yVal = 5)
##ax4 = plt.subplot2grid((2,2),(1,1))
##plt.plot(T,Y[:,3], label='3')
##plotEthogram(ax4, T, etho, alpha=0.5, yVal = 5)
##plt.ylabel('tica')
##plt.show()
#x,y = Y.T[:2]
#ax = plt.subplot(111)
#multicolor(ax,x,y,None,etho[:,0],cm.viridis, threedim = 0)
##plt.plot( Y[:,0],  Y[:,1], 'ro')
#plt.show()
##print len(velo), len(Y[:,1])
##for i in range(2):
##    fig = plt.figure('Correlates{}'.format(i),(3.5,7))
##    plt.subplot(311)
##    plt.scatter(velo, Y[:,i], label='pc1', alpha=0.01)
##    
##    plt.subplot(313)
##    plt.scatter(turns, Y[:,i], label='pc1', alpha=0.01)
##
##
##plt.show()
#
#######################################
##
## neural dimensionality tica
##
######################################
#fig = plt.figure('Manifolds!',(7,7))
#ax = fig.gca(projection='3d')
#x,y,z = Y.T[:3]
#multicolor(ax,x,y,z,etho[:,0],cm.viridis)
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")
#plt.show()
#######################################
##
## Working with Eigenworms
##
######################################
#
#if eigenwormplot:
#    fig = plt.figure('Behavior',(7,7))
#    alpha = 0.2
#    ax1 = plt.subplot(511)
#    plotEthogram(ax1, T, etho, alpha, yVal = max(pc12[:,0]))
#    plt.plot(T, velo, 'k')
#    
#    ax2 = plt.subplot(512, sharex=ax1)
#    plotEthogram(ax1, T, etho, alpha, yVal = max(pc12[:,0]))
#    plt.plot(T, pc12[:,0], 'k')
#    plt.ylabel('Eigenworm 1')
#    
#    ax3 = plt.subplot(513, sharex=ax1)
#    plotEthogram(ax1, T, etho, alpha, yVal = max(pc12[:,1]))
#    plt.plot(T, pc12[:,1], 'k')
#    plt.ylabel('Eigenworm 2')
#    
#    ax4 = plt.subplot(514, sharex=ax1)
#    plotEthogram(ax1, T, etho, alpha, yVal = max(pc3))
#    plt.plot(T, pc3, 'k')
#    plt.ylabel('Eigenworm 3')
#    
#    ax5 = plt.subplot(515, sharex=ax1)
#    cax1 = plotHeatmap(T, Y)
#    plt.xlabel('Time (s)')
#    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#    fig.colorbar(cax1, cax=cbar_ax)
#    fig.subplots_adjust(left = 0.1, bottom = 0.05, top =0.95, right=0.8, hspace=0.15)
#    plt.show()
#
#######################################
##
## PCA of neural data
##
######################################
##%%
#nComp = 5
#
##neuroSmooth -=np.mean(neuroSmooth, axis=0)
##neuroSmooth = Y - np.mean(Y, axis=0)
#pca = PCA(n_components = nComp)
#pcs = pca.fit_transform(neuroSmooth)
#print pcs.shape
## order by weight
#indices = np.arange(lenY)
#indices = np.argsort(pcs[:,0])
#pcs = pcs[indices]
#
#fig = plt.figure('PCA',(7,7))
#plt.subplot(221)
#cax1 = plotHeatmap(T, neuroSmooth[indices])
#plt.xlabel('Time (s)')
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(cax1, cax=cbar_ax)
#fig.subplots_adjust(left = 0.1, bottom = 0.05, top =0.95, right=0.8, hspace=0.15)
#
## plot the weights
#plt.subplot(222)
#rank = np.arange(0, len(pcs))
#
##weightsum = np.concatenate([np.zeros((lenY,1)),np.cumsum(pcs, axis=1)], axis=1)
##for i in range(4):
##    plt.fill_betweenx(rank, weightsum[:,i], weightsum[:,i+1], step='pre')
#plt.fill_betweenx(rank, np.zeros(lenY),pcs[:,0], step='pre')
#plt.fill_betweenx(rank, np.zeros(lenY),pcs[:,1], step='pre')
#
#
#ax3 = plt.subplot(223)
#plt.fill_between(np.arange(nComp),pca.explained_variance_ratio_)
#ax3.step(np.arange(nComp),np.cumsum(pca.explained_variance_ratio_), where = 'pre')
##ax3.set_ylabel('Explained variance')
#
#ax4 = plt.subplot(224)
#for i in range(4):
#    ax4.plot(T,0.1*i+pca.components_[i], label=i)
#plotEthogram(ax4, T, etho, alpha=0.5, yVal = 0.5)
#
#plt.show()
##%%
########################################
###
### correlate neural data and behavior
###
#######################################
##fig = plt.figure('CorrelatesFwd',(7,7))
##plt.subplot(311)
##plt.scatter(pc12[:,0], pca.components_[0], label='pc1', alpha=0.01)
##plt.subplot(312)
##plt.scatter(pc12[:,0], pca.components_[1], label='pc2', alpha=0.01)
##plt.subplot(313)
##plt.scatter(pc12[:,0], pca.components_[2], label='pc3', alpha=0.01)
##
##fig = plt.figure('Correlatesbwd',(7,7))
##plt.subplot(311)
##plt.scatter(pc12[:,1], pca.components_[0], label='pc1', alpha=0.01)
##plt.subplot(312)
##plt.scatter(pc12[:,1], pca.components_[1], label='pc2', alpha=0.01)
##plt.subplot(313)
##plt.scatter(pc12[:,1], pca.components_[2], label='pc3', alpha=0.01)
##
##
##fig = plt.figure('CorrelatesTurn',(7,7))
##plt.subplot(311)
##plt.scatter(turns, pca.components_[0], label='pc1', alpha=0.01)
##plt.subplot(312)
##plt.scatter(turns, pca.components_[1], label='pc2', alpha=0.01)
##plt.subplot(313)
##plt.scatter(turns, pca.components_[2], label='pc3', alpha=0.01)
#
##%%
#######################################
##
## neural dimensionality
##
######################################
#fig = plt.figure('Manifolds!',(7,7))
#ax = fig.gca(projection='3d')
#x,y,z = pca.components_[:3]
#multicolor(ax,x,y,z,etho[:,0],cm.viridis)
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")
#
#
##%%
#######################################
##
## reproject neural dynamics in lower dim space
##
######################################
##%%
#a  = pca.inverse_transform(pcs)
#plt.subplot(211)
#cax1 = plotHeatmap(T, neuroSmooth[indices])
#plt.subplot(212)
#cax1 = plotHeatmap(T, a)
#
#plt.show()
#
#######################################
##
## projection on behavioral axes
##
######################################
