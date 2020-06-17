import numpy as np
from toolbox import CTViewer_Multiax as cvm

filename = "./detection_vision/train/LKDS-00005_detresult2.npy"
volimg = np.load(filename)
cvm.view_CT(volimg)
