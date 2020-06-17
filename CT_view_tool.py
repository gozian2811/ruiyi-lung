import PyQt5
import PIL
import random
import csv
import os
import matplotlib
matplotlib.use('TkAgg')
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt
import math as m
from skimage import measure, morphology
from copy import deepcopy
import skimage.io as io
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from toolbox import MITools as mt
from toolbox import CTViewer_Multiax as cvm

def fiji_like():
	def fileopen():
		filename = filedialog.askopenfilename(filetypes=[('numpy files', '*.npy'), ('mhd files', '*.mhd'), ('dicom files', '*.dcm'), ('all', '*')])
		#filename = "E:/tianchi_project/TIANCHI_examples/train_1/LKDS-00001.mhd"
		if filename=='':
			return
		print (repr(filename))
		prename, extname = os.path.splitext(filename)
		if extname=='.mhd':
			full_image_info = sitk.ReadImage(filename)
			full_scan = sitk.GetArrayFromImage(full_image_info)
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]
			volimg, new_spacing = mt.resample(full_scan, old_spacing)
		elif extname=='.dcm':
			pathname = os.path.split(filename)[0]
			full_scan, full_image_info, patientid = mt.read_sph_scan(pathname)
			#cvm.view_CT(full_scan)
			old_spacing = np.array(full_image_info.GetSpacing())[::-1]
			volimg, new_spacing = mt.resample(full_scan, old_spacing)
		elif extname=='.npy':
			volimg = np.load(filename)
		else:
			print ('unknown data type')
			return
		label = tk.Label(tool_view, image=cvm.view_CT(volimg))
		label.pack()
		if 'full_scan' in dir():
			del full_scan
		del volimg
		tool_view.quit()
	tool_view = tk.Tk()
	canvas = tk.Canvas(tool_view)
	menubar = tk.Menu(tool_view)
	menubar.add_command(label="Open", command=fileopen)
	menubar.add_command(label="Exit", command=tool_view.quit)
	tool_view.config(menu=menubar)
	tool_view.protocol("WM_DELETE_WINDOW", tool_view.quit)
	tool_view.mainloop()

fiji_like()
