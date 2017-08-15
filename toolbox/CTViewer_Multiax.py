import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys

class CTViewer:
    def __init__(self, volume, ax, direction, slicenum = 0):
        self.valid = False
        self.volume = volume
        self.slicenum = slicenum
        self.direction = direction
        self.ax = ax
        self.cid_scroll = ax.figure.canvas.mpl_connect('scroll_event', self)
        ax.figure.canvas.mpl_connect('key_press_event',self)
        ax.figure.canvas.mpl_connect('axes_enter_event',self.__enter__)
        ax.figure.canvas.mpl_connect('axes_leave_event',self.__leave__)
        ax.figure.canvas.mpl_connect('button_press_event',self.__dblclick__)
        if self.direction == 'z':
            self.zaxis = self.slicenum
            self.maxslice = self.volume.shape[0]
        elif self.direction == 'x':
            self.xaxis = self.slicenum
            self.maxslice = self.volume.shape[1]
        elif self.direction == 'y':
            self.yaxis = self.slicenum
            self.maxslice = self.volume.shape[2]
        self.my_label = 'slice:%d/%d' %(self.slicenum+1, self.maxslice)
        self.ax.set_title(self.my_label)
                
    def __enter__(self,event):
        if event.inaxes == self.ax:
            self.valid = True

    def __leave__(self,event):
        if event.inaxes == self.ax:
            self.valid = False

    def __dblclick__(self,event):
        if event.dblclick and event.inaxes == self.ax:
            new_fig = plt.figure(figsize=(6,6))
            new_ax = new_fig.add_subplot(111)
            if self.direction == 'z':
                self.zaxis = self.slicenum
                new_ax.imshow(self.volume[self.zaxis], plt.cm.gray)
            elif self.direction == 'x':
                self.xaxis = self.slicenum
                new_ax.imshow(self.volume[:,self.xaxis,:], plt.cm.gray)
            elif self.direction == 'y':
                self.yaxis = self.slicenum
                new_ax.imshow(self.volume[:,:,self.volume.shape[2]-1-self.yaxis], cmap=plt.cm.gray)
            CTViewer(self.volume, new_ax, self.direction, self.slicenum)
            plt.show()
            plt.close()

    def __call__(self, event):
        if self.valid == False:
            return
        if event.name == 'scroll_event' and event.button == 'up' or event.name == 'key_press_event' and event.key == 'up':
            if self.slicenum < self.maxslice-1:
                self.slicenum += 1
        elif event.name == 'scroll_event' and event.button == 'down' or event.name == 'key_press_event' and event.key == 'down':
            if self.slicenum > 0:
                self.slicenum -= 1
        self.my_label = 'slice:%d/%d' %(self.slicenum+1, self.maxslice)
        self.ax.cla()
        self.ax.set_title(self.my_label)
        if self.direction == 'z':
            self.zaxis = self.slicenum
            self.ax.imshow(self.volume[self.zaxis], plt.cm.gray)
        elif self.direction == 'x':
            self.xaxis = self.slicenum
            self.ax.imshow(self.volume[:,self.xaxis,:], plt.cm.gray)
        elif self.direction == 'y':
            self.yaxis = self.slicenum
            self.ax.imshow(self.volume[:,:,self.volume.shape[2]-1-self.yaxis], cmap=plt.cm.gray)
        plt.draw()
                
def view_CT(volume):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(131)
    ax.imshow(volume[0], plt.cm.gray)
    CTViewer(volume, ax, 'z')
    bx = fig.add_subplot(132)
    bx.imshow(volume[:,0,:],plt.cm.gray)
    CTViewer(volume, bx, 'x')
    cx = fig.add_subplot(133)
    cx.imshow(volume[:,:,volume.shape[2]-1],plt.cm.gray)
    CTViewer(volume,cx, 'y')
    plt.show()
    plt.close()

def view_coordinations(volume, candidate_coords = [], window_size = 40, reverse = True, slicewise = False, show = True):
    half_window = int(window_size/2)
    volume_regioned = np.ndarray(shape=volume.shape, dtype=volume.dtype)
    volume_regioned[:,:,:] = volume
    for coord in candidate_coords:
        if reverse:
            coord = coord[::-1]
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
