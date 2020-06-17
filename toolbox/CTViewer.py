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