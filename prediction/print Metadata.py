# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 17:11:28 2018
write out a table of all datasets with metainformation.
@author: monika
"""
import dataHandler as dh
import numpy as np
import pprint
import os
from prediction import userTracker
################################################
#
# grab all the data we will need
#
################################################
strains = {'AML32':'GCaMP6s',
              'AML18':'GFP',
              'AML175':'GFP',
              'AML70':'GCaMP6s',
              'Special': 'GCaMP6s'
    }
bg = {'AML32':'wt',
              'AML18':'wt',
              'AML175':'lite-1',
              'AML70':'lite-1',
              'Special': 'check'
    }
    
conditions = {'moving': 'freely moving',
              'chip': 'freely moving ',
              'transition': 'transiently paralyzed with tetramisole',
              'immobilized': 'nanobeads',
              }
plate = {'moving': 'imaging plate',
              'chip': 'microfluidic ',
              'transition': 'microfluidic',
              'immobilized': 'agarose pad',
              }
              

def dump_keys(d, lvl=0):
    for k, v in d.iteritems():
        print '%s%s' % (lvl * ' ', k)
        if type(v) == dict:
            dump_keys(v, lvl+1)

data = {}
print "Condition, Strain, Unique Identifier,	Indicator,Background,Arena,Duration (min), Number of Neurons "

for typ in ['Special', 'AML32', 'AML18', 'AML175', 'AML70']:
    for condition in ['transition','immobilized','moving', 'chip']:
        path = userTracker.dataPath()
        folder = os.path.join(path, '{}_{}/'.format(typ, condition))
        dataLog = os.path.join(path,'{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition))
        outLoc = os.path.join(path, 'Analysis/{}_{}_results.hdf5'.format(typ, condition))
        outLocData = os.path.join(path,'/Analysis/{}_{}.hdf5'.format(typ, condition))
        #print strains[typ], conditions[condition]
        try:
            # load multiple datasets
            dataSets = dh.loadDictFromHDF(outLocData)
            keyList = np.sort(dataSets.keys())
            results = dh.loadDictFromHDF(outLoc) 
            # store in dictionary by typ and condition
            idn = '{}_{}'.format(typ, condition)
            
            for ki, key in enumerate(keyList):
                if idn!='AML70_moving':
                    print "{}, {},{}, {}, {}, {}, {}, {}".format(conditions[condition],typ, key, strains[typ],bg[typ], plate[condition],  int(dataSets[key]['Neurons']['Activity'].shape[1]/6./60.),dataSets[key]['Neurons']['Activity'].shape[0])
                #if idn == 'AML32_moving' and ki==0:
                #    print dump_keys(results[key])
                    
                    #pprint.pprint(dataSets[key], depth=1)
        except IOError:
            #print typ, condition , 'not found.'
            pass
print 'Done reading data.'


