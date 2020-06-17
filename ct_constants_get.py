from tqdm import tqdm
from copy import copy
import os
import glob
import numpy as np

train_sets = ["subset0", "subset1", "subset2", "subset3", "subset4", "subset5", "subset6", "subset7", "subset8", "subset9"]
#train_sets = ["subset0"]
data_dir = "tianchi_project/luna_cubes_56_overbound"
pfiles = []
nfiles = []
for set in train_sets:
	train_dir = os.path.join(data_dir, set)
	pdir = os.path.join(train_dir,"npy","*.npy")
	pfiles.extend(glob.glob(pdir))
	#ndir = os.path.join(train_dir,"npy_non","*.npy")
	#nfiles.extend(glob.glob(ndir))

#tfiles = copy(pfiles)
#tfiles.extend(nfiles)

maxvalue = -2000
minvalue = 2000
for fileenum in enumerate(tqdm(pfiles)):
	tfile = fileenum[1]
	volume = np.load(tfile)
	vmax = volume.max()
	vmin = volume.min()
	if maxvalue<vmax:
		maxvalue = vmax
		maxfile = tfile
	if minvalue>vmin:
		minvalue = vmin
print("maxvalue:%d minvalue:%d" %(maxvalue, minvalue))
print("%s" %(maxfile))
