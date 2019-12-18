### noiseModel.py
# We suspect that the neural dynamics we observe in freely moving GCaMP worms have three components:
# 1) Neural Signals representing locomotion
# 2) Neural signals representing other biological processes (sensory processing, working memory, etc)
# 3) Noise, including from motion artifact
#
# We know for sure that (1) & (3) exist. We can decode locomotion so  we know that (1) must be persent.
# And we know that noise  (3) is present  because we see it in the GFP moving animals.
#
# To bolster our argument that neural signals representing other biological processes are present in our recordings,
# we will attempt to reject the following null hypothesis:
#
# Null hypothesis is that ONLY neural signals representing locomotion AND noise are present. E.g. only (1) and (3) and
# not (2).
#
#
# We will build uop our null model by taking a real freely moving GFP worm that therefore has only noise (3) and we
# will synthetically generate pure locmootory signals (1) and adjust the relative proportion of noise to
# locomotory signals.
#
# We  will study the predicitive performance of our decoder on this model and also the percent variance explaiend by the
# locomotory siganl to try to assess whether the null model fits our experimental observations.


## Find a good GFP recording to use.

dataset = '/Users/leifer/workspace/PredictionCode/AML18_moving/BrainScanner20160506_160928_MS/heatData.mat'

# Import the recording based on the output of Jeff's 3dBrain matlab analysis pipeline
import scipy.io as sio
mat_contents = sio.loadmat(dataset)
rPhotoCorr = mat_contents['rPhotoCorr']
rRaw = mat_contents['rRaw']
gPhotoCorr = mat_contents['gPhotoCorr']
gRaw = mat_contents['gRaw']
Ratio2 = mat_contents['Ratio2']

# Import Monika's ICA'd version (with my modification s.t. there is no z-scoring)
# (Note this section is copy pasted from S2.py)
import prediction.dataHandler as dh
import numpy as np
data = {}
for typ in ['AML18','AML32']:
    for condition in ['moving']:
        folder = '/Users/leifer/workspace/PredictionCode/{}_{}/'.format(typ, condition)
        dataLog = '/Users/leifer/workspace/PredictionCode/{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
        outLoc = "/Users/leifer/workspace/PredictionCode/Analysis/{}_{}_results.hdf5".format(typ, condition)
        outLocData = "/Users/leifer/workspace/PredictionCode/Analysis/{}_{}.hdf5".format(typ, condition)

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
            print typ, condition, 'not found.'
            pass
print 'Done reading data.'




import matplotlib.pyplot as plt
from prediction import provenance as prov


ordIndx=data['AML18_moving']['input']['BrainScanner20160506_160928']['Neurons']['ordering']

fig1=plt.figure()
plt.subplot(4, 1, 1)
plt.imshow(rPhotoCorr[ordIndx,:],aspect='auto')
plt.colorbar()
plt.title('rPhotoCorr')

plt.subplot(4,1,2)
plt.imshow(gPhotoCorr[ordIndx,:],aspect='auto')
plt.colorbar()
plt.title('gPhotoCorr')

plt.subplot(4,1,3)
plt.imshow(np.divide(gPhotoCorr,rPhotoCorr)[ordIndx,:],aspect='auto')
plt.colorbar()
plt.title('gPhotoCorr/rPhotoCorr')


ax=plt.subplot(4, 1, 4)
plt.imshow(data['AML18_moving']['input']['BrainScanner20160506_160928']['Neurons']['ActivityFull'],aspect='auto')
plt.colorbar()
prov.stamp(ax,0,-.3)
plt.title('"ActivityFull" (ICAd, plus Andys modified normalization) \n' +dataset)
plt.show()



fig2=plt.figure()
ax21=plt.subplot(5, 1, 1)
plt.imshow(rRaw[ordIndx, :], aspect='auto', vmin=np.nanmean(rRaw)-2*np.nanstd(rRaw), vmax=np.nanmean(rRaw)+2*np.nanstd(rRaw))
plt.colorbar()
plt.title('rRaw')


ax22=plt.subplot(5,1,2)
plt.plot(10*np.sum(np.isnan(rRaw) ,axis=0)) #plot how many nans at each time point (x10 fo visibility)
plt.plot(np.nanmean(rRaw, axis=0)) # plot the average rRaw intesnity
plt.title('rRaw Average Intensity, and number of neurons with NaN')
ax22.set_ylim( np.nanmean(rRaw)-1*np.nanstd(rRaw), np.nanmean(rRaw)+2*np.nanstd(rRaw) )
ax22.set_xlim(0,rRaw.shape[1])
pos21 = ax21.get_position()
pos22 = ax22.get_position()
ax22.set_position([pos21.x0,pos22.y0,pos21.width,pos22.height])



plt.subplot(5,1,3)
plt.imshow(gRaw[ordIndx, :], aspect='auto', vmin=np.nanmean(gRaw)-2*np.nanstd(gRaw), vmax=np.nanmean(gRaw)+2*np.nanstd(gRaw))
plt.colorbar()
plt.title('gRaw')

plt.subplot(5,1,4)
plt.imshow(np.divide(gRaw, rRaw)[ordIndx, :], aspect='auto')
plt.colorbar()
plt.title('gRaw/rRaw')


ax=plt.subplot(5, 1, 5)
plt.imshow(data['AML18_moving']['input']['BrainScanner20160506_160928']['Neurons']['ActivityFull'],aspect='auto')
plt.colorbar()
prov.stamp(ax,0,-.3)
plt.title('"ActivityFull" (ICAd, plus Andys modified normalization) \n' +dataset)
plt.xlabel('Volumes (6 vol /s )')
plt.show()






## Plot velocity and body curvature
vel=data['AML18_moving']['input']['BrainScanner20160506_160928']['Behavior']['AngleVelocity']
bodycurv= data['AML18_moving']['input']['BrainScanner20160506_160928']['Behavior']['Eigenworm3']

if True:
    plt.subplots(4,1,sharex=True)
    ax1=plt.subplot(4, 1, 1)
    plt.imshow(rPhotoCorr[ordIndx,:],aspect='auto')
    plt.colorbar()
    plt.title('rPhotoCorr')

    ax2=plt.subplot(4,1,2)
    plt.imshow(gPhotoCorr[ordIndx,:],aspect='auto')
    plt.colorbar()
    plt.title('gPhotoCorr')



    ax3=plt.subplot(4,1,3)
    plt.plot(vel)
    plt.title('Velocity')
    ax3.set_xlim(ax1.get_xlim())

    ax4=plt.subplot(4, 1, 4)
    plt.plot(bodycurv)
    prov.stamp(ax4,0,-.3)
    plt.title('Body Curvature')
    ax4.set_xlim(ax1.get_xlim())

    # align the axes
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    pos4 = ax4.get_position()
    ax3.set_position([pos1.x0,pos3.y0,pos1.width,pos3.height])
    ax4.set_position([pos1.x0,pos4.y0,pos1.width,pos4.height])

    plt.show()



if False: #Just checking the difference between TimeFull and Time
    plt.figure()
    plt.plot(data['AML32_moving']['input']['BrainScanner20170610_105634']['Neurons']['TimeFull'], label='TimeFull')
    plt.plot(data['AML32_moving']['input']['BrainScanner20170610_105634']['Neurons']['Time'],label='Time')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(np.diff(data['AML32_moving']['input']['BrainScanner20170610_105634']['Neurons']['TimeFull']),
             label='TimeFull')
    plt.plot(np.diff(data['AML32_moving']['input']['BrainScanner20170610_105634']['Neurons']['Time']),
             label='Time')
    plt.legend()
    plt.show()

## Extract neural weights learned by the SLM from a GCaMP Recording



## Shuffle the neural Weights

## Generate synthetic pure locomotory signals based on the GFP recording

## Combine the GFP and synthetic locomotory signals for a given relative strength (lambda)

## Figure out how normalization should occur.  [Has somethign to do with lambda.]

## Measure Decoder Performance (R^2)

## Measure variance explained

#Get the heatmap values

