import os
import numpy as np

def annostr2coord(annostr, handomit=True):
	'''
	if type(annostr)==unicode:
		if annostr.find(u'*')>=0:
			return None
		coordbegin = -1
		coordend = -1
		for ci in range(len(annostr)):
			if coordbegin<0:
				if annostr[ci]>=u'0' and annostr[ci]<=u'9':
					coordbegin = ci
			elif (annostr[ci]<u'0' or annostr[ci]>u'9') and annostr[ci]!=u' ':
				coordend = ci
				break
		if coordbegin>=0:
			if coordend<0:
				coordend = len(annostr)
			coordstr = annostr[coordbegin:coordend]
			annotation = np.array(coordstr.split(u' '), dtype=int)
		else:
			annotation = None
	'''
	if type(annostr)==str:
		if handomit and annostr.find('*')>=0:
			return None
		coordbegin = -1
		coordend = -1
		for ci in range(len(annostr)):
			if coordbegin<0:
				if annostr[ci]>='0' and annostr[ci]<='9':
					coordbegin = ci
			elif (annostr[ci]<'0' or annostr[ci]>'9') and annostr[ci]!=' ':
				coordend = ci
				break
		if coordbegin>=0:
			# annotation = np.array(annostr.split('（')[0].split(' '), dtype=int)
			if coordend<0:
				coordend = len(annostr)
			coordstr = annostr[coordbegin:coordend]
			coords = coordstr.split(' ')
			if len(coords)==3:
				annotation = np.array(coordstr.split(' '), dtype=int)
			else:
				annotation = None
		else:
			annotation = None
	else:
		annotation = None
	return annotation


def sample_pathology_label(filepath, mode='binclass'):
	#the parameter 'mode' lies in {'binclass', 'ternclass'}.
	#if the mode is 'regression', the parameter 'malignant_positive' is of no use.
	if mode=='binclass':
		label_dict = {'AAH':0, 'AIS':0, 'MIA':1}
	elif mode=='ternclass':
		label_dict = {'AAH':0, 'AIS':1, 'MIA':2}
	else:
		print('label mode incorrect in sample_malignancy_label(.)')
		exit()
	filename = os.path.basename(filepath)
	label = filename.split('_')[2]
	
	return label_dict[label]
