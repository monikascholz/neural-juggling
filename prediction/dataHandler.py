# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:41:10 2017
load and compare multiple data sets for behavior prediction.
@author: monika scholz
"""
from __future__ import division # make division give you floats, as is standard for python3
import scipy.io
import os
import numpy as np
import matplotlib.pylab as plt
import scipy.interpolate
#from scipy.signal import medfilt, deconvolve
from skimage.transform import resize
from sklearn import preprocessing
import makePlots as mp
from scipy.ndimage.filters import gaussian_filter1d
import h5py
from scipy.special import erf
#
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,  FastICA


def recrWorm(av, turns, thetaTrue, r, show = 0):
    """recalculate eigenworm prefactor from angular velocity and turns."""
    thetaCum = np.cumsum(np.copy(av)/6.) # division due to rate: it's velocity per secod instread of per volume
    # reset theta every minute to real value
    dt = np.arange(50, len(thetaCum), 60)
    for tt in dt:    
        thetaCum[tt:] -= -thetaTrue[tt]+thetaCum[tt]
    
    radius = np.zeros(len(av)) 
    for tt in dt:
        radius[tt:] = r[tt]
#    plt.plot(radius)
#    plt.show()
    r = radius
#    thetaCum -= thetaCum[50]-thetaTrue[50] 
#    thetaCum -= np.mean(thetaCum)- np.mean(thetaTrue)
    #tt = 0
    #thetaCum[tt:] -= -thetaTrue[tt]+thetaCum[tt]
    # recalculate the phase angle from cumulative phase
    theta = np.mod(thetaCum, 2*np.pi)#-np.pi
    sign = np.ones(len(theta))
    sign[np.where(np.abs(theta-np.pi)>np.pi/2.)] = -1
    # do the same for the true angle
    thetaD = np.mod(thetaTrue, 2*np.pi)
    thetaDS = np.where(np.abs(thetaD-np.pi)>np.pi/2., -thetaD, thetaD)
    if show:
        plt.figure('Real and reconstructed phase angle')
        plt.subplot(221)
        plt.plot(thetaCum, label = 'reconstructed')
        plt.plot(thetaTrue, label = 'real')
        
        #plt.scatter(thetaCum,thetaTrue, label = 'reconstructed')
        #plt.plot(thetaTrue, label = 'real')
        plt.ylabel('Accumulated phase angle')
        plt.legend()
        plt.subplot(222)
        plt.plot(thetaCum-thetaTrue, label = 'residuals')
        plt.plot(np.cumsum(thetaCum-thetaTrue), label = 'cumulative residuals')
        plt.ylabel('Phase difference')
        plt.legend()
        plt.subplot(223)
        plt.plot(theta, label = 'reconstructed')
        plt.plot(thetaD, label ='real')#-np.pi)
        plt.ylabel('Phase angle (rad)')
        plt.xlabel('Time (Volumes)')
        plt.legend()
        plt.subplot(224)
        plt.scatter(sign*theta,thetaDS,s=1, alpha=0.1)
        plt.ylabel('Phase angle (rad)')
        plt.xlabel('reconstructed Phase angle (rad)')
        plt.tight_layout()
        plt.show(block=True)
    # recalculate eigenworms
    x = -np.sqrt(r)*np.tan(sign*theta)/np.sqrt((np.tan(sign*theta)**2+1))
    y = -sign*np.sqrt(r-x**2)
    return x,y, turns

def deconvolveCalcium(Y, show=False):
    """deconvolve with GCamp6s response digitized from Nature volume 499, pages 295–300 (18 July 2013)
        doi:10.1038/nature12354"""
    # fit function -- fitted with least squares from digitized data
    pars =  [ 0.38036106 , 0.00565365 , 1.00621729 , 0.31627363 ]
    def fitfunc(x,A,m, tau1, s):
        return A*erf((x-m)/s)*np.exp(-x/tau1)
    gcampXN = np.linspace(0,Y.shape[1]/6., Y.shape[1])
    gcampYN = fitfunc(gcampXN, *pars)
    Ydec = np.real(np.fft.ifft(np.fft.fft(Y, axis = 1)/np.fft.fft(gcampYN)))*np.sum(gcampYN)
    if show:
        plt.subplot(221)
        plt.plot(gcampX, gcampY)
        plt.plot(gcampXN[:18], gcampYN[:18])
        ax = plt.subplot(222)
        frq, psGC = calcFFT(gcampYN, time_step=1/6.)
        plt.plot(frq, psGC)
        ax.set_yscale('log',nonposy='clip')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel("Power spectrum")

        Ydec = []
        
        #     line by line fft of neural signal
        frq, fft = calcFFT(Y, time_step=1/6.)
        for line in fft:
            plt.plot(frq, line, 'r', alpha=0.1)

        ax.set_yscale('log',nonposy='clip')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel("Power spectrum")
        plt.show()
        vmax, vmin=1,0
        ax = plt.subplot(223)
        cax1 = ax.imshow(Y, aspect='auto', interpolation='none', origin='lower',vmax=vmax, vmin=vmin)
        ax = plt.subplot(224)
        pcax1 = ax.imshow(Ydec, aspect='auto', interpolation='none', origin='lower',vmax=vmax, vmin=vmin)
        plt.show()
    return Ydec

def calcFFT(data, time_step=1/6.):
    """plot frequency of data"""
    fft = []
    if len(data.shape)>1:
        for line in data:
            ps = np.abs(np.fft.fft(line))**2
            freqs = np.fft.fftfreq(line.size, time_step)
            idx = np.argsort(freqs)
            fft.append(ps[idx])
    else:
        ps = np.abs(np.fft.fft(data))**2
        freqs = np.fft.fftfreq(data.size, time_step)
        idx = np.argsort(freqs)
        fft = ps[idx]
    return freqs[idx], fft

def makeEthogram(anglevelocity, pc3):
    """use rotated Eigenworms to create a new Ethogram."""
    etho = np.zeros((len(anglevelocity),1))
    # set forward and backward
    etho[np.where(anglevelocity>0.05)] = 1
    etho[np.where(anglevelocity<=-0.05)] = -1
    # overwrite this in case of turns
    etho[np.abs(pc3)>10] = 2
    return etho

def loadPoints(folder,  straight = True):
    """get tracked points from Pointfile."""
    points = np.squeeze(scipy.io.loadmat(os.path.join(folder,'pointStatsNew.mat'))['pointStatsNew'])
    
    if straight:
        return [p[1] for p in points]
    else:
        return [p[2] for p in points]
    
def loadEigenBasis(filename, nComp=3, new = True):
    """load the specific worm basis set."""
    if new:
        eigenworms = np.loadtxt(filename)[:nComp]
    else:
        eigenworms = scipy.io.loadmat(filename)['eigbasis'][:nComp]
    # ncomponents controls how many we use
    eigenworms = resize(eigenworms, (nComp,99))
    return eigenworms


def estimateEigenwormError(folder, eigenworms, show=False):
    """use the high resolution behavior to get a variance estimate.
    This will be wrong or meaningless if the centerlines were copied between frames."""
    # calculate centerline projections for full movie
    clFull, clIndices = loadCenterlines(folder, full = True)
    print 'done loading'
    pcsFull, meanAngle, lengths, refPoint = calculateEigenwormsFromCL(clFull, eigenworms)
    print 'done projecting'
    # split array by indices into blocks corresponding to volumes
    pcsSplit = np.split(pcsFull, clIndices, axis=1)
    # calculate standard deviation and mean
    pcsM = np.array([np.nanmean(p, axis=1) for p in pcsSplit]).T
    pcsErr = np.array([np.nanstd(p, axis=1) for p in pcsSplit]).T
    #length = np.array([len(p[0]) for p in pcs]).T
    #
    if show:
        i=2 # which eigenworm
        plt.figure('Eigenworm error')
        plt.subplot(211)
        plt.plot(pcsFull[i][clIndices], label='discret eigenworms')
        plt.plot(pcsM[i], label='averaged eigenworms', color='r')
        plt.fill_between(range(len(pcsM[i])), pcsM[i]-pcsErr[i], pcsM[i]+pcsErr[i], alpha=0.5, color='r')
        plt.subplot(212)
        m, err = np.sort(pcsM[i]), pcsErr[i][np.argsort(pcsM[i])]
        plt.plot(np.sort(pcsFull[i][clIndices]), label='discret eigenworms')
        plt.plot(m, label='averaged eigenworms', color='r')
        plt.fill_between(range(len(pcsM[i])), m-err, m+err, alpha=0.5, color='r')
        plt.show()
    return pcsM, pcsErr, pcsFull[:,clIndices.astype(int)], pcsFull

def calculateEigenwormsFromCL(cl, eigenworms):
    """takes (x,y) pairs from centerlines and returns eigenworm coefficients."""
    # coordinate segments
    diffVec = np.diff(cl, axis=1)
    # calculate tangential vectors
    wcNew = np.unwrap(np.arctan2(-diffVec[:,:,1], diffVec[:,:,0]))
    #################these things are needed for reconstruction
    # get mean angle
    meanAngle = np.mean(wcNew, axis=1)
    # get segment lengths
    lengths = np.sqrt(diffVec[:,:,1]**2+diffVec[:,:,0]**2)
    # get overall alignment in space
    # reference point to start with
    refPoint = cl[:,0]
    # calculate mean subtracted tangent angles
    wcNew = wcNew-meanAngle[:, np.newaxis]
    # project onto Eigenworms
    pcsNew = np.dot(eigenworms,wcNew.T)
    return pcsNew, meanAngle, lengths, refPoint

def calculateCLfromEW(pcsNew, eigenworms, meanAngle, lengths, refPoint):
    """takes eigenworms and a few reference parameters to recreate centerline."""
    # now we recreate the worm
    wcR = np.dot(pcsNew.T, eigenworms) + meanAngle[:, np.newaxis] 
    # recreate tangent vectors with correct length
    tVecs = np.stack([lengths*np.cos(wcR), -lengths*np.sin(wcR)], axis=2)
    # start at same point as original CL
    clApprox = refPoint[:, np.newaxis] + np.cumsum(tVecs, axis=1)
    return clApprox       

def loadCenterlines(folder, full = False, wormcentered = False):
    """get centerlines from centerline.mat file"""
    #cl = scipy.io.loadmat(folder+'centerline.mat')['centerline']
    tmp =scipy.io.loadmat(os.path.join(folder,'centerline.mat'))
    
    cl = np.rollaxis(scipy.io.loadmat(os.path.join(folder,'centerline.mat'))['centerline'], 2,0)
    if wormcentered:
        cl = np.rollaxis(scipy.io.loadmat(os.path.join(folder,'centerline.mat'))['wormcentered'], 1,0)

    tmp = scipy.io.loadmat(os.path.join(folder,'heatDataMS.mat'))
    
    clTime = np.squeeze(tmp['clTime']) # 50Hz centerline times
    volTime =  np.squeeze(tmp['hasPointsTime'])# 6 vol/sec neuron times
    
    clIndices = np.rint(np.interp(volTime, clTime, np.arange(len(clTime))))  
    if not full:
        # reduce to volume time
        cl = cl[clIndices.astype(int)]
    #wcNew = wc[clIndices.astype(int)]
    #epNew = ep[clIndices.astype(int)]
    
#    for cl in clNew[::10]:
#        plt.plot(cl[:,0], cl[:,1])
#    plt.show()
    return cl ,clIndices.astype(int)
    
def transformEigenworms(pcs, dataPars):
    """interpolate, smooth Eigenworms and calculate associated metrics like velocity."""
    pc3, pc2, pc1 = pcs
    #mask nans in eigenworms by linear interpolation
    for pcindex, pc in enumerate(pcs):
        # mask nans by linearly interpolating
        mask = np.isnan(pc1)

        if np.any(mask):
            pc[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pc1[~mask])
            import warnings
            warnings.warn('Found ' + str(np.sum(mask)) + 'NaNs that are being interpolated over in the behavior PC1, PC2 or PC3.')
        pcs[pcindex] = pc
        
    theta = np.unwrap(np.arctan2(pcs[2], pcs[1]))
    # convolution with gaussian kernel derivative
    velo = gaussian_filter1d(theta, dataPars['gaussWindow'], order=1)
    # velo is in radians/frame
    for pcindex, pc in enumerate(pcs):
        pcs[pcindex] = gaussian_filter1d(pc, dataPars['medianWindow'])
    return pcs, velo, theta

def decorrelateNeuronsICA(R, G):
    """use PCA to remove covariance in Green and Red signals."""
    Ynew = []
    ica = FastICA(n_components = 2)
    for li in range(len(R)):
        Y = np.vstack([R[li], G[li]]).T
        sclar2= StandardScaler(copy=True, with_mean=True, with_std=True)
        Y = sclar2.fit_transform(Y)
        S = ica.fit_transform(Y)
        # order components by max correlation with red signal
        v = [np.corrcoef(s,R[li])[0,1] for s in S.T]
        idn = np.argmin(np.abs(v))
        # check if signal needs to be inverted
        sign = np.sign(np.corrcoef(S[:,idn],G[li])[0,1])
        signal = sign*(S[:,idn])
        Ynew.append(signal)
    return np.array(Ynew)#, np.mean(var, axis=0), Rs, Gs 
    
def decorrelateNeurons(R, G):
    """use PCA to remove covariance in Green and Red signals."""
    Ynew = []
    var = []
    Rs,Gs=[], []
    pca = PCA(n_components = 2)
    for li in range(len(R)):
        Y = np.vstack([R[li], G[li]]).T
        sclar2= StandardScaler(copy=True, with_mean=True, with_std=True)
        Y = sclar2.fit_transform(Y)
        compFull = pca.fit_transform(Y)
        pcs = pca.components_
        Ynew.append(compFull[:,1])
        
        var.append(pca.explained_variance_ratio_)
        Yhat = np.dot(compFull[:,1:],pcs[1:,:])
        Yhat += sclar2.mean_
        Rs.append(Yhat[:,0])
        Gs.append(Yhat[:,1])
    return np.array(Ynew), np.mean(var, axis=0), np.array(Rs), np.array(Gs)

def preprocessNeuralData(R, G, dataPars):
    """zscore etc for neural data."""
    # prep neural data by masking nans
    mask = np.isnan(R)
    R[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), R[~mask])
    mask = np.isnan(G)
    G[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), G[~mask])
    
    # smooth with GCamp6 halftime = 1s
    RS =np.array([gaussian_filter1d(line,dataPars['windowGCamp']) for line in R])       
    GS =np.array([gaussian_filter1d(line,dataPars['windowGCamp']) for line in G])
    print("windowGCaMP:", str(dataPars['windowGCamp']))
#    YR = GS/RS
    
##    meansubtract = False#False#True
##    if meansubtract:
##        # long-window size smoothing filter to subtract overall fluctuation in SNR
##        wind = 90
##        mean = np.mean(rolling_window(np.mean(YR,axis=0), window=2*wind), axis=1)
##        #pad with normal mean in front to correctly center the mean values
##        mean = np.pad(mean, (wind,0), mode='constant', constant_values=(np.mean(np.mean(YR,axis=0)[:wind])))[:-wind]
##        # do the same in the end
##        mean[-wind:] = np.repeat(np.mean(np.mean(YR,axis=0)[:-wind]), wind)
##        YN = YR-mean
##    else:
#        YN = YR
    #YN, _,GS,RS = decorrelateNeurons(RS, GS)
    YN = decorrelateNeuronsICA(R, G)
    YN = np.array([gaussian_filter1d(line,dataPars['windowGCamp']) for line in YN])
    #$YN = GS/RS
    # percentile scale
    R0 = np.percentile(YN, [20], axis=1).T
    dR = np.divide(YN-R0,np.abs(R0))
    #dR = YN
    # zscore values 
    YN =  preprocessing.scale(YN.T).T
    R0 = np.percentile(GS/RS, [20], axis=1).T
    RM = np.divide(GS/RS-R0,np.abs(R0))
#    plt.imshow(dR, aspect='auto')
#    plt.show()
    return YN, dR, GS, RS, RM

def loadData(folder, dataPars, ew=1):
    """load matlab data."""
    print 'Loading ', folder
    try:
        data = scipy.io.loadmat(os.path.join(folder,'heatDataMS.mat'))
    except IOError:
        print 'IOERROR'
        data = scipy.io.loadmat(os.path.join(folder,'heatData.mat'))
    # unpack behavior variables
    ethoOrig, xPos, yPos, vel, pc12, pc3 = data['behavior'][0][0].T
    # get centerlines with full temporal resolution of 50Hz
    clFull, clIndices = loadCenterlines(folder, full=True)
    # load new eigenworms
    eigenworms = loadEigenBasis(filename = '../utility/Eigenworms.dat', nComp=3, new=True)
    # get full set of Eigenworms
    pcsFull, meanAngle, lengths, refPoint = calculateEigenwormsFromCL(clFull, eigenworms)
    # do Eigenworm transformations and calculate velocity etc. 
    pcs, velo, theta = transformEigenworms(pcsFull, dataPars)
    #downsample to 6 volumes/sec
    pc3, pc2, pc1 = pcs[:,clIndices]

    velo = velo[clIndices]*50. # to get it in per Volume units -> This is radians per sec
    
    theta = theta[clIndices] # get it in per seconds
    cl = clFull[clIndices]
    # ethogram redone
    etho = makeEthogram(velo, pc3)
    #etho = np.squeeze(ethoOrig)
    # mask nans in ethogram
    ethomask = np.isnan(etho)
    if np.any(ethomask):
        etho[ethomask] = 0

    rRaw=np.array(data['rRaw'])[:,:len(np.array(data['hasPointsTime']))]
    gRaw=np.array(data['gRaw'])[:,:len(np.array(data['hasPointsTime']))]

    rphotocorr = np.array(data['rPhotoCorr'])[:, :len(np.array(data['hasPointsTime']))]
    gphotocorr = np.array(data['gPhotoCorr'])[:, :len(np.array(data['hasPointsTime']))]

    #load neural data
    debug = True
    original = False
    if original:
        R = np.copy(rphotocorr)
        G = np.copy(gphotocorr)
    else:
        R = rRaw.copy()
        G = gRaw.copy()
        vps = dataPars['volumeAcquisitionRate']

        #remove outliers with a sliding window of 40 volumes (or 6 seconds)
        R = nanOutliers(R, np.round(2*3.3*vps).astype(int))[0]
        G = nanOutliers(G, np.round(2*3.3*vps).astype(int))[0]

        R = correctPhotobleaching(R, vps)
        G = correctPhotobleaching(G, vps)

        if debug:

            nplots=2
            chosen = np.round(np.random.rand(nplots)*R.shape[0]).astype(int)
            import matplotlib.pyplot as plt
            plt.cla()
            plt.subplot(nplots, 1, 1,)
            plt.plot(G[chosen[0],:],'g', label='gRaw w/ python photocorrection')
            plt.plot(gphotocorr[chosen[0],:],'b', label='gPhotoCorr via matlab')
            plt.title('Comparsion of Green Channel photocorrection methods for neuron ' + str(chosen[0]))
            plt.legend()

            plt.subplot(nplots,1,2)
            plt.plot(R[chosen[0], :], 'r', label='rRaw w/ python photocorrection')
            plt.plot(rphotocorr[chosen[0], :], 'm', label='rPhotoCorr via matlab')
            plt.title('Comparsion of Red Channel photocorrection methods for neuron ' + str(chosen[0]))
            plt.legend()
            plt.show()






    #
    Ratio = np.array(data['Ratio2'])[:,:len(np.array(data['hasPointsTime']))]
    Y, dR, GS, RS, RM = preprocessNeuralData(R, G, dataPars)
    try:
        dY = np.array(data['Ratio2D']).T
    except KeyError:
        dY = np.zeros(Y.shape)
    order = np.array(data['cgIdx']).T[0]-1
    # read flagged neurons
    try:
        if len(data['flagged_neurons'])>0:
            badNeurs = np.array(data['flagged_neurons'][0])
            order = np.delete(order, badNeurs)
    except KeyError:
        pass
    # get rid of predominantly nan neurons
    #fracNans = np.sum(np.isnan(Ratio), axis=1)/1.0/len(Ratio[0])
    
    #order = order[np.where(fracNans<0.1)]
    #lets interpolate small gaps but throw out larger gaps.
    # make a map with all nans smoothed out if larger than some window    
    nanmask =[np.repeat(np.nanmean(chunky_window(line, window= dataPars['interpolateNans']), axis=1), dataPars['interpolateNans']) for line in Ratio]
    nanmask = np.array(nanmask)[:,:Y.shape[1]]
    if 'flagged_volumes' in data.keys():
        if len(data['flagged_volumes'])>0:
            print data['flagged_volumes']
            nanmask[:,np.array(data['flagged_volumes'][0])] = np.nan
    Rfull = np.copy(Y)
    Rfull[np.isnan(nanmask)] =np.nan
    
    Y = Y[order]
    dR = dR[order]
    RM = RM[order]
    #deconvolved data
    YD = deconvolveCalcium(Y)
    #regularized derivative
    dY = dY[order]
    # store relevant indices -- crop out the long gaps of nans adn flagged timepoints
    nonNan  = np.where(np.any(np.isfinite(nanmask),axis=0))[0]
    
    # create a time axis in seconds
    T = np.arange(Y.shape[1])/6.
    # redo time axis in seconds for nan issues
    T = np.arange(Y[:,nonNan].shape[1])/6.
    # 
    time = np.squeeze(data['hasPointsTime'])
    time -= time[nonNan[0]]
    
    # unpack neuron position (only one frame, randomly chosen)
    try:
        neuroPos = data['XYZcoord'][order].T
    except KeyError:
        neuroPos = []
        print 'No neuron positions:', folder
    
    
    # create a dictionary structure of these data
    dataDict = {}
    # store centerlines subsampled to volumes
    dataDict['CL'] = cl[nonNan]
    dataDict['goodVolumes'] = nonNan
    dataDict['Behavior'] = {}
    print RM.shape
    tmpData = [vel[:,0], pc1, pc2, pc3, velo, theta, etho, xPos, yPos]
    for kindex, key in enumerate(['CMSVelocity', 'Eigenworm1', 'Eigenworm2', \
    'Eigenworm3',\
                'AngleVelocity','Theta', 'Ethogram', 'X', 'Y']):
        dataDict['Behavior'][key] = tmpData[kindex][nonNan]
    dataDict['Behavior']['EthogramFull'] = etho
    dataDict['Neurons'] = {}
    dataDict['Neurons']['Indices'] =  T#np.arange(Y[:,nonNan].shape[1])/6.#T[nonNan]
    dataDict['Neurons']['Time'] =  time[nonNan] # actual time
    dataDict['Neurons']['TimeFull'] =  time # actual time
    dataDict['Neurons']['ActivityFull'] =  Rfull[order] # full activity
    dataDict['Neurons']['Activity'] = preprocessing.scale(Y[:,nonNan].T).T # redo because nans
    dataDict['Neurons']['RawActivity'] = dR[:,nonNan]
    dataDict['Neurons']['derivActivity'] = dY[:,nonNan]
    dataDict['Neurons']['deconvolvedActivity'] = YD[:,nonNan]
    dataDict['Neurons']['RedRaw'] = RS
    dataDict['Neurons']['Ratio'] = RM[:,nonNan]
    dataDict['Neurons']['GreenRaw'] = GS
    dataDict['Neurons']['Positions'] = neuroPos
    dataDict['Neurons']['ordering'] = order
    dataDict['Neurons']['valid'] = nonNan
    dataDict['Neurons']['orientation'] = 1 # dorsal or ventral
    
    return dataDict
    
    
def loadMultipleDatasets(dataLog, pathTemplate, dataPars, nDatasets = None):
    """load matlab files containing brainscanner data. 
    string dataLog: file containing Brainscanner names with timestamps e.g. BrainScanner20160413_133747.
    path pathtemplate: relative or absoluet location of the dataset with a formatter replacing the folder name. e.g.
                        GoldStandardDatasets/{}_linkcopy

    return: dict of dictionaries with neuron and behavior data
    """
    datasets={}
    for lindex, line in enumerate(np.loadtxt(dataLog, dtype=str, ndmin = 2)[:nDatasets]):
        folder = ''.join([pathTemplate, line[0], '_MS'])
        datasets[line[0]] = loadData(folder, dataPars)
    return datasets

def loadNeuronPositions(filename):
    x = scipy.io.loadmat(filename)['x']
    y = scipy.io.loadmat(filename)['y']
    z = scipy.io.loadmat(filename)['z']
    neuronID = scipy.io.loadmat(filename)['ID']
    # remove non-head neurons
    indices = np.where((y<-2.3)&(x<0.1))
    return np.stack((neuronID[indices],x[indices],y[indices],z[indices]))

def rankTransform(neuroMap):
    """takes a matrix and transforms values into rank within the colum. ie. neural dynamics: for each neuron
    calculate its rank at the current time."""
    temp = neuroMap.argsort(axis=0)
    rank = temp.argsort(axis=0)
    return rank


def rolling_window(a, window):
    a = np.pad(a, (0,window), mode="constant", constant_values=(np.nan,))
    shape = a.shape[:-1] + (a.shape[-1] - window, window)
    strides = a.strides + (a.strides[-1],)
    
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
def chunky_window(a, window):
    xp =  np.r_[a, np.nan + np.zeros((-len(a) % window,))]
    return xp.reshape(-1, window)
    
def saveDictToHDF(filePath, d):
    f = h5py.File(filePath,'w')
    for fnKey in d.keys(): #this level is datasets ie., Brainscanner0000000
        for amKey in d[fnKey].keys():# this level is analysis type ie., PCA
            if type(d[fnKey][amKey]) is not dict:
                 dataPath = '/%s/%s'%(fnKey,amKey)
                 f.create_dataset(dataPath,data=d[fnKey][amKey])
            else:
                for attKey in d[fnKey][amKey].keys(): # This level is entry ie. PCAweights
                    if type(d[fnKey][amKey][attKey]) is not dict:
                        dataPath = '/%s/%s/%s'%(fnKey,amKey,attKey)
                        f.create_dataset(dataPath,data=d[fnKey][amKey][attKey])
                    else:
                        for bKey in d[fnKey][amKey][attKey].keys():
                            
                            dataPath = '/%s/%s/%s/%s'%(fnKey,amKey,attKey,bKey)
                            f.create_dataset(dataPath,data=d[fnKey][amKey][attKey][bKey])
    f.close()
    return

def loadDictFromHDF(filePath):
    f = h5py.File(filePath,'r')
    d = {}
    for fnKey in f.keys():
        d[fnKey] = {}
        for amKey in f[fnKey].keys():
            if isinstance(f[fnKey][amKey], h5py.Dataset):
                d[fnKey][amKey] = f[fnKey][amKey][...]
            else:
                d[fnKey][amKey] = {}
                for attKey in f[fnKey][amKey].keys():
                    if isinstance(f[fnKey][amKey][attKey], h5py.Dataset):
                        d[fnKey][amKey][attKey] = f[fnKey][amKey][attKey][...]
                    else:
                        d[fnKey][amKey][attKey] = {}
                        for bKey in f[fnKey][amKey][attKey].keys():
                            d[fnKey][amKey][attKey][bKey] = f[fnKey][amKey][attKey][bKey][...]
                        
    f.close()
    return d

def correctPhotobleaching(raw, vps=6):
    """ Apply photobleaching correction to raw signals, works on a whole heatmap of neurons

    Use Xiaowen's photobleaching correction method which fits an exponential and then divides it off
    Note we are following along with the explicit noise model from Xiaowen's Phys Rev E paper Eq 1 page 3
    Chen, Randi, Leifer and Bialek, Phys Rev E 2019

    takes raw neural signals N (rows) neruons x T (cols) time points
    we require that each neuron is a row and that there are more time points than rows

    """

    # Make sure the data we are getting makes sense.
    assert raw.ndim <= 2, "raw fluorescent traces passed to correctPhotobleaching() have the wrong number of dimensions"

    if raw.ndim == 2:
        assert raw.shape[1] > raw.shape[
            0], "raw fluorescent traces passed to correctPhotobleaching() have the wrong size! Perhaps it is transposed?"

    window_s = 12.6  # time of median filter window in seconds
    medfilt_window = np.int(np.ceil(window_s * vps / 2.) * 2 + 1)  # make it an odd number of frames

    photoCorr = np.zeros_like(raw)  # initialize photoCorr

    debug = False

    performMedfilt  = True



    if raw.ndim == 2:
        if performMedfilt:
            # perform median filtering (Nan friendly)
            from scipy.signal import medfilt
            smoothed = medfilt(raw, [1, medfilt_window])
        else:
            smoothed = raw.copy()

        N_neurons = raw.shape[0]

        # initialize the exponential fit parameter outputs
        popt = np.zeros([N_neurons, 3])
        pcov = np.zeros([3, 3, N_neurons])

        # Exponential Fit on each neuron
        for row in np.arange(N_neurons):
            popt[row, :], pcov[:, :, row], xVals = fitPhotobleaching(smoothed[row, :], vps)
            photoCorr[row, :] = popt[row, 0] * raw[row, :] / expfunc(xVals, *popt[row, :])

            showPlots = False
            if debug:
                if np.random.rand() > 0.85:
                    showPlots = True

            if showPlots:
                import matplotlib.pyplot as plt
                plt.plot(xVals, raw[row, :], 'b-', label=['raw, row: '+np.str(row)])
                plt.plot(xVals, photoCorr[row, :], "r-",
                         label=['photoCorr, row: '+np.str(row)])
                plt.xlabel('Time (s)')
                plt.ylabel('ActivityTrace')
                plt.title('correctPhotobleaching()')
                plt.legend()
                plt.show()

    elif raw.ndim == 1:
        smoothed = medfilt(raw, medfilt_window)
        popt, pcov, xVals = fitPhotobleaching(smoothed, vps)
        photoCorr= popt[0] * raw / expfunc(xVals, *popt)

    else:
        raise ValueError('The  number dimensions of the data in correctPhotobleaching are not 1 or 2')

    return photoCorr



def fitPhotobleaching(activityTrace, vps):
    """"
    Fit an exponential.

    Accepts a single neuron's activity trace.

    Use Xiaowen's photobleaching correction method which fits an exponential and then divides it off
    Note we are following along with the explicit noise model from Xiaowen's Phys Rev E paper Eq 1 page 3
    Chen, Randi, Leifer and Bialek, Phys Rev E 2019


    """
    assert activityTrace.ndim == 1, "fitPhotobleaching only works on single neuron traces"

    # Now we will fit an exponential, following along with the tuotiral here:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    xVals = np.array(np.arange(activityTrace.shape[-1])) / np.float(vps)

    # Identify just the data that is not a NaN
    nonNaNs = np.logical_not(np.isnan(activityTrace))

    from scipy.optimize import curve_fit

    # set up some bounds on our exponential fitting parameter, y=a*exp(bx)+c
    num_recordingLengths = 8 # the timescale of exponential decay should not be more than eight recording lengths long
    # because otherwise we could have recorded for way longer!!

    #Scale the activity trace to somethign around one.
    scaleFactor=np.nanmean(activityTrace)
    activityTrace=activityTrace/scaleFactor

    bounds = ([0, 1 / (num_recordingLengths * np.nanmax(xVals)), 0],  # lower bounds
              [np.nanmax(activityTrace[nonNaNs])*1.5, 0.5, 2 * np.nanmean(activityTrace)])  # upper bound

    # as a first guess, a is half the max, b is 1/(half the length of the recording), and c is the average
    popt_guess = [np.nanmax(activityTrace) / 2, 2 / np.nanmax(xVals), np.nanmean(activityTrace)]

    popt, pcov = curve_fit(expfunc, xVals[nonNaNs], activityTrace[nonNaNs], p0=popt_guess,
                           bounds=bounds)
    debug = False
    if np.logical_and(np.random.rand() > 0.9, debug):
        showPlots = True
    else:
        showPlots = False

    if showPlots:
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot(xVals, activityTrace, 'b-', label='data')
        plt.plot(xVals, expfunc(xVals, *popt), "r-",
                 label='fit a*exp(-b*x)+c: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.xlabel('Time (s)')
        plt.ylabel('ActivityTrace')
        plt.legend()

    ## Now we want to inspect our fit, find values that are clear outliers, and refit while excluding those
    residual = activityTrace - expfunc(xVals, *popt)  # its ok to have nan's here

    nSigmas = 3  # we want to exclude points that are three standard deviations away from the fit

    # Make a new mask that excludes the outliers
    excOutliers = np.copy(nonNaNs)
    excOutliers[(np.abs(residual) > (nSigmas * np.nanstd(residual)))] = False

    # Refit excluding the outliers, use the previous fit as initial guess
    # note we relax the bounds here a bit

    try:
        popt, pcov = curve_fit(expfunc, xVals[excOutliers], activityTrace[excOutliers], p0=popt,bounds=bounds)
    except:
        popt, pcov = curve_fit(expfunc, xVals[excOutliers], activityTrace[excOutliers], p0=popt)

    if showPlots:
        import matplotlib.pyplot as plt
        plt.plot(xVals, expfunc(xVals, *popt), 'y-',
                 label='excluding outliers fit a*exp(-b*x)+c: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        plt.legend()

    #rescale the amplitude, a, back to full size
    popt[0]=scaleFactor*popt[0]

    #rescale the c parameter
    popt[2]=scaleFactor*popt[2]
    return popt, pcov, xVals


def expfunc(x, a, b, c):
    # type: (xVals, a, b, c) -> yVals
    return a * np.exp(-b * x) + c





from skimage.util import view_as_windows

def nanOutliers(data, window_size, n_sigmas=3):
    """using a Hampel filter to detect outliers and replace them with nans.

        This is based on Xiaowen Chen's MATLAB routine for preprocessing from her Phys Rev E 2019 paper.

    The algorithm assumes the underlying series should be Gaussian. This could be changed by changing the k parameter.
    Data can contain nans, these will be ignored for median calculation.
    data: (M, N) array of M timeseries with N samples each.
    windowsize: integer. This is half of the typical implementation.
    n_sigmas: float or integer
    """
    # deal with N = 1 timeseries to conform to (M=1,N) shape
    if len(data.shape)<2:
        data = np.reshape(data, (1,-1))
    k = 1.4826 # scale factor for Gaussian distribution
    M, N = data.shape # store data shape for quick use below
    # pad array to achieve at least the same size as original
    paddata = np.pad(data.copy(), pad_width= [(0,0),(window_size//2,window_size//2)], \
                     constant_values=(np.median(data)), mode = 'constant')
    # we use strides to create rolling windows. Not nice in memory, but good in performance.
    # because its meant for images, it creates some empty axes of size 1.
    tmpdata = view_as_windows(paddata, (1,window_size))
    # crop data to center
    tmpdata = tmpdata[:M, :N]
    x0N = np.nanmedian(tmpdata, axis = (-1, -2))
    s0N =k * np.nanmedian(np.abs(tmpdata - x0N[:,:,np.newaxis, np.newaxis]), axis = (-1,-2))
    # hampel condition
    hampel = (np.abs(data-x0N) - (n_sigmas*s0N))
    indices = np.where(hampel>0)
    # cast to float to allow nans in output data
    newData = np.array(data, dtype=float)
    newData[indices] = np.nan

    Debug = False
    if Debug:
        chosen = np.squeeze( np.round(np.random.rand(1) * M-1).astype(int))
        import matplotlib.pyplot as plt
        plt.cla()
        plt.plot(data[chosen, :],'r')
        plt.plot(newData[chosen, :],'b')
        plt.title('nanOutlier(), neuron: ' +str(chosen))
        plt.show()

    return newData, indices




