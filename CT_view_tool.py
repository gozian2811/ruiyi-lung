import random
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt
import math as m
from skimage import measure, morphology
from copy import deepcopy
import skimage.io as io
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import PyQt5
import PIL
import tkinter as tk
from toolbox import MITools as mt
from toolbox import CTViewer as cv

def fiji_like():
    tool_view = tk.Tk()
    canvas = tk.Canvas(tool_view)
    '''
    class CTViewer:
        def __init__(self, volume, ax,  zaxis=0):
            self.volume = volume
            self.zaxis = zaxis
            self.ax = ax
            self.tool_view = tkinter.Tk()
            self.frame = tkinter.Frame(self.tool_view)
            self.frame.pack()
            self.canvas = tkinter.Canvas(self.frame)
            self.cid = ax.figure.canvas.mpl_connect('scroll_event', self)
            self.tool_view.mainloop()
            print('slice:%d/%d' % (self.zaxis + 1, self.volume.shape[0]))

        def __call__(self, event):
            if event.button == 'up':
                if self.zaxis < self.volume.shape[0] - 1:
                    self.zaxis += 1
                    print('slice:%d/%d' % (self.zaxis + 1, self.volume.shape[0]))
            elif event.button == 'down':
                if self.zaxis > 0:
                    self.zaxis -= 1
                    print('slice:%d/%d' % (self.zaxis + 1, self.volume.shape[0]))
            self.ax.cla()
            #self.ax.imshow(self.volume[self.zaxis], plt.cm.gray)
            self.img = PIL.Image.fromarray(self.volume[self.zaxis], mode='L')
            self.imgTk = ImageTk.PhotoImage(self.img)
            #self.canvas = FigureCanvasTkAgg(self.imgTk, master=tool_view)
            self.w = Canvas(self.imgTk)
            self.w.pack()
            self.tool_view.update()
            self.tool_view.mainloop()
            self.canvas.draw()
            #plt.draw()
    def view_CT(volume):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.imshow(volume[0], plt.cm.gray)
        #canvas = FigureCanvas(volume[0])
        img = PIL.Image.fromarray(volume[0], mode='L')
        imgTk = ImageTk.PhotoImage(img)
        w = Canvas(imgTk)
        w.pack()
        #canvas = FigureCanvasTkAgg(imgTk, master=tool_view)
        CTViewer(volume, ax)
        canvas.show()
        #plt.show()
        '''
    def fileopen():
        filename = tk.filedialog.askopenfilename(filetypes=[('mhd', '*.mhd')])
        #filename = "E:/tianchi_project/TIANCHI_examples/train_1/LKDS-00001.mhd"
        print (repr(filename))
        full_image_info = sitk.ReadImage(filename)
        full_scan = sitk.GetArrayFromImage(full_image_info)
        old_spacing = np.array(full_image_info.GetSpacing())[::-1]
        img2, new_spacing = mt.resample(full_scan, old_spacing)
        label = tk.Label(tool_view, image=cv.view_CT(img2))
        label.pack()
        canvas.pack()
    menubar = tk.Menu(tool_view)
    menubar.add_command(label="Open", command=fileopen)
    menubar.add_command(label="Exit", command=tool_view.quit)
    tool_view.config(menu=menubar)
    tool_view.mainloop()
fiji_like()
