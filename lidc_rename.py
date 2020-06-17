import os
import shutil
import dicom
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from toolbox import BasicTools as bt
from toolbox import MITools as mt
from toolbox import LIDCTools as lt

if __name__ == "__main__":
	sourceset = '/home/fyl/programs/lung_project/data_samples/lidc_cubes_64_overbound_ipris/npy'
	targetsets = ['/home/fyl/datasets/luna_64/train', '/home/fyl/datasets/luna_64/test']
	#targetsets = ['/home/fyl/programs/lung_project/data_samples/lidc_cubes_64_overbound_ipris/npy']
	dataset = './DOI'
	patients = lt.retrieve_scan(dataset)
	sources = glob(sourceset + '/*_annotations.npy')
	print(len(sources))
	for targetset in targetsets:
		samples = glob(targetset + '/*_annotations.npy')
		print(len(samples))
		for sample in enumerate(tqdm(samples)):
			sample = sample[1]
			pathname = os.path.dirname(sample)
			samplename = os.path.basename(sample)
			seperation = samplename.find('_')
			data_name = samplename[:seperation]
			for patient in patients:
				if patient.find(data_name)>=0:
					patient_id = patient.split('/')[-3]
					serie_uid = patient.split('/')[-1]
					if data_name==serie_uid:
						break
			orisamplename = patient_id + samplename[seperation:]
			oripathname = sourceset + '/' + orisamplename
			shutil.copyfile(oripathname, sample)
			#samplerename = patient_id + samplename[seperation:]
			#repathname = pathname + '/' + samplerename
			#os.rename(sample, repathname)