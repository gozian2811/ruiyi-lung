import os.path
import numpy as np
import SimpleITK as sitk
import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from toolbox import MITools as mt
from toolbox import Lung_Cluster as lc
from toolbox import CTViewer as cv

def lung_slic_tool(filename):
	prename, extname = os.path.splitext(filename)
	pathname = os.path.split(filename)[0]

	if (extname == '.npy'):
		segresult = np.load(filename)
		#labels = np.load(prename+'.label.npy')
		cv.view_CT(segresult)
		
	elif (extname == '.mhd'):
		npyname = filename+'.npy'
		filelist = [pathname+'/'+i for i in os.listdir(pathname)]
		if npyname in filelist:
			segresult = np.load(npyname)
			#labels = np.load(filename+'.label.npy')
		else:
			full_image_info = sitk.ReadImage(filename)
			full_scan = sitk.GetArrayFromImage(full_image_info)
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]
			image, new_spacing = mt.resample(full_scan, old_spacing)
			#####
			#image = image[195:200]
			#####
			print("slic segmenting")
			labels = lc.slic_segment(image, num_segments=500, compactness=0.001, result_output=True, view_result=True)
			print("segmentation complete")
			#np.save(filename+'.npy',segresult)
			#np.save(filename+'.label.npy',labels)
	#lc.view_segment(lc.normalization(image), labels)
        

if __name__ == '__main__':
	filename = tkinter.filedialog.askopenfilename(filetypes=[('mhd', '*.mhd'),('numpy files','*.npy')])
	#filename = "TIANCHI_data/train/LKDS-00001.mhd"
	lung_slic_tool(filename)