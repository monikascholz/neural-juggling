#!/usr/bin/env python

"""detect which user is using the machine and change the path to the data accordingly."""
import socket

def dataPath():
	"""hardcoded or automatic detection of the users' data path."""
	host = socket.gethostname()
	if host== 'subsitute for andys computer name':
		path = '/Users/leifer/workspace/PredictionCode/'
	elif host=='nif3004':
		path = '/media/scholz_la/hd2/Data/Data'
	else:
		raise Exception('This is not a known host. Edit util/userTracker.py to add your hostname and data paths.')
	return path