import numpy as np
from CTViewer import view_CT

filename = "nodule_cubes/train/npy_random/LKDS-00249_cc_0_random.npy"
volume = np.load(filename)
view_CT(volume)
