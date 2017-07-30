import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys

class CTViewer:
	def __init__(self, volume, ax, zaxis = 0):
                
		self.volume = volume
		self.zaxis = zaxis
		self.ax = ax
		self.cid = ax.figure.canvas.mpl_connect('scroll_event', self)
		self.cidkey = ax.figure.canvas.mpl_connect('key_press_event',self)
		self.my_label = 'slice:%d/%d' %(self.zaxis+1, self.volume.shape[0])
		plt.title(self.my_label)
		#print(self.my_label)

	def __call__(self, event):
		if event.name == 'scroll_event' and event.button == 'up' or event.name == 'key_press_event' and event.key == 'up':
			if self.zaxis < self.volume.shape[0]-1:
				self.zaxis += 1
				#print('slice:%d/%d' %(self.zaxis+1, self.volume.shape[0]))
		elif event.name == 'scroll_event' and event.button == 'down' or event.name == 'key_press_event' and event.key == 'down':
			if self.zaxis > 0:
				self.zaxis -= 1
				#print('slice:%d/%d' %(self.zaxis+1, self.volume.shape[0]))
		self.my_label = 'slice:%d/%d' %(self.zaxis+1, self.volume.shape[0])
		self.ax.cla()
		plt.title(self.my_label)
		self.ax.imshow(self.volume[self.zaxis], cmap=plt.cm.gray)
		plt.draw()

def view_CT(volume):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(volume[0], plt.cm.gray)
	CTViewer(volume, ax)
	plt.show()

def view_coordinations(volume, candidate_coords = [], window_size = 40, reverse = True, slicewise = False, show = True):
	half_window = int(window_size/2)
	volume_regioned = np.ndarray(shape=volume.shape, dtype=volume.dtype)
	volume_regioned[:,:,:] = volume
	for coord in candidate_coords:
		if reverse:
			coord = coord[::-1]		#reverse the coordination order from x,y,z to z,y,x
		#adjust the bound
		bottombound = coord - np.array([half_window, half_window, half_window], dtype=int)
		topbound = coord + np.array([half_window, half_window, half_window], dtype=int)
		for i in range(len(volume.shape)):
			if bottombound[i]<0:
				bottombound[i] = 0
			elif bottombound[i]>=volume.shape[i]:
				bottombound[i] = volume.shape[i] - 1
			if topbound[i]<0:
				topbound[i] = 0
			elif topbound[i]>=volume.shape[i]:
				topbound[i] = volume.shape[i] - 1
		#draw a rectangular bound around the candidate position
		if slicewise:
			volume_regioned[coord[0],bottombound[1]:topbound[1],bottombound[2]] = 500
			volume_regioned[coord[0],bottombound[1]:topbound[1],topbound[2]] = 500
			volume_regioned[coord[0],bottombound[1],bottombound[2]:topbound[2]] = 500
			volume_regioned[coord[0],topbound[1],bottombound[2]:topbound[2]] = 500
		else:
			volume_regioned[bottombound[0]:topbound[0],bottombound[1]:topbound[1],bottombound[2]] = 500
			volume_regioned[bottombound[0]:topbound[0],bottombound[1]:topbound[1],topbound[2]] = 500
			volume_regioned[bottombound[0]:topbound[0],bottombound[1],bottombound[2]:topbound[2]] = 500
			volume_regioned[bottombound[0]:topbound[0],topbound[1],bottombound[2]:topbound[2]] = 500
	if show:	
		view_CT(volume_regioned)
	
	return volume_regioned