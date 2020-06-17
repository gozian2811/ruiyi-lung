import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import csv
import random
import scipy.ndimage
from glob import glob
from skimage import measure, morphology
from copy import deepcopy
from . import MITools as mt
from . import CTViewer as cv

def readimage(img_path):
    itk_img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(itk_img)  # indexs are z y x,the axis is -> x,\v y (0,0) is the left top
    shape=img_array.shape
    return img_array


def split(image):
    shape=image.shape
    az=int(shape[0]/2)
    ax=int(shape[1]/4)
    bound=shape[1]-1
    if(image[az][1][1]<-2000):
        bi=np.array(image<-2000,dtype=np.int8)
        image[bi==1]=-1024

    for y in range(shape[2]-1,1,-1):
        if image[az][y][ax]>=-320&image[az][y-10][ax]>=-320:
            bound=y
            break

    for z in range(shape[0]):
        for y in range(shape[1]):
            if(y >=bound):
                for x in range(shape[2]):
                    image[z][y][x]=-1024

    return image

def through(image):
    shape = image.shape
    for zt in range(image.shape[0]):
        for yt in range(image.shape[1]):
            image[zt,yt,0] = image[0,0,0]
            image[zt,yt,-1] = image[0,0,0]
        for xt in range(image.shape[2]):
            image[zt,0,xt] = image[0,0,0]
            image[zt,-1,xt] = image[0,0,0]
    return image

def candidate_detection(segimage,flag=None):
    shape=segimage.shape
    NoduleMatrix = np.zeros(shape,dtype=int)
    for i in range(shape[0]):
        if flag is None or flag[i]==1:
            for j in range(shape[1]):
                index=np.where(segimage[i][j]!=1024)[0]
                for k in index:
                    #Do judge
                    if(segimage[i][j][k]>-600):
                        NoduleMatrix[i][j][k]=1
                        #order z,y,x
    index = np.where(NoduleMatrix == 1)
    return NoduleMatrix, index

def cluster(index,scale,iterate=False):
    def square_distance(x,y):
        sqdist = (x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2])
        return sqdist
    def distance(x,y):
        dis = m.sqrt((x[0]-y[0])*(x[0]-y[0]) + (x[1]-y[1])*(x[1]-y[1]) + (x[2]-y[2])*(x[2]-y[2]))
        return dis
    def nearssqdist(candidatej,new,scale):
        snew = [new[0]*candidatej[1], new[1]*candidatej[1], new[2]*candidatej[1]]
        ssqdist = (candidatej[0][0]-snew[0])*(candidatej[0][0]-snew[0]) + (candidatej[0][1]-snew[1])*(candidatej[0][1]-snew[1]) + (candidatej[0][2]-snew[2])*(candidatej[0][2]-snew[2])
        if ssqdist<candidatej[1]*candidatej[1]*scale*scale:
            return ssqdist
        else:
            return -1
    def add(candidatej,new):
        newcluster = [[],[],[]]
        newcluster[1] = [candidatej[1][0]+new[0], candidatej[1][1]+new[1], candidatej[1][2]+new[2]]
        newcluster[2] = candidatej[2] + 1
        newcluster[0] = [newcluster[1][0]/newcluster[2], newcluster[1][1]/newcluster[2], newcluster[1][2]/newcluster[2]]
        return newcluster
    def subtract(candidatej,old):
        oldcluster = [[],[],[]]
        oldcluster[1] = [candidatej[1][0]-old[0], candidatej[1][1]-old[1], candidatej[1][2]-old[2]]
        oldcluster[2] = candidatej[2] - 1
        oldcluster[0] = [oldcluster[1][0]/oldcluster[2], oldcluster[1][1]/oldcluster[2], oldcluster[1][2]/oldcluster[2]]
        return oldcluster
    def fastadd(candidatej,new):
        newcluster = [[],[]]
        newcluster[0] = [candidatej[0][0]+new[0], candidatej[0][1]+new[1], candidatej[0][2]+new[2]]
        newcluster[1] = candidatej[1] + 1
        return newcluster
    def fastsubtract(candidatej,old):
        oldcluster = [[],[]]
        oldcluster[0] = [candidatej[0][0]-old[0], candidatej[0][1]-old[1], candidatej[0][2]-old[2]]
        oldcluster[1] = candidatej[1] - 1
        return oldcluster

    print("Clustering:")
    positionz = index[0]
    positiony = index[1]
    positionx = index[2]
    l=len(positionx)
    ##Here we need to do a cluster to find the candidate nodules
    candidate=[]
    if l==0:
        return candidate
    center_index_cluster = 0 - np.ones(len(positionx), dtype=int)
    point = [float(positionx[l-1]), float(positiony[l-1]), float(positionz[l-1])]#point is a list
    center_index_cluster[l-1] = 0
    #candidate.append([point,point, 1])
    candidate.append([point, 1])
    for i in range(l-1):
        point=[float(positionx[i]), float(positiony[i]), float(positionz[i])] #The current point to be clustered
        nearsqdist = scale*scale
        nearcand = -1
        #find the older cluster
        for j in range(len(candidate)):
            ssqdist = nearssqdist(candidate[j],point,scale)
            if ssqdist>=0 and ssqdist<nearsqdist*candidate[j][1]*candidate[j][1]:
                nearsqdist = ssqdist/(candidate[j][1]*candidate[j][1])
                nearcand = j
            '''
            sqdist = square_distance(point,candidate[j][0])
            if sqdist<scale*scale and sqdist<nearsqdist: #which means we should add the point into this cluster
                #Notice the type that candidate is a list so we need to write a demo
                nearsqdist = sqdist
                nearcand = j
            '''
        if nearcand>0:
            candidate[nearcand] = fastadd(candidate[nearcand], point)
            center_index_cluster[i] = nearcand
        else: #create a new cluster
            candidate.append([point, 1])
            #candidate.append([point, point, 1])
    iternum = 0
    if iterate:
        converge = False
        while not converge:
            print("iteration:%d" %(iternum+1))
            iternum += 1
            converge = True
            for i in range(l):
                point=[float(positionx[i]), float(positiony[i]), float(positionz[i])] #The current point to be clustered
                flag=0
                nearsqdist = scale
                nearcand = -1
                #find the older cluster
                for j in range(len(candidate)):
                    if candidate[j][1]<=0:
                        continue
                    ssqdist = nearssqdist(candidate[j],point,scale)
                    if ssqdist>=0 and ssqdist<nearsqdist*candidate[j][1]*candidate[j][1]:
                        nearsqdist = ssqdist/(candidate[j][1]*candidate[j][1])
                        nearcand = j
                if nearcand>0 and nearcand!=center_index_cluster[i]:
                    #print("i:%d center:%d" %(i, center_index_cluster[i]))
                    converge = False
                    if center_index_cluster[i]>=0:
                        candidate[center_index_cluster[i]] = fastsubtract(candidate[center_index_cluster[i]], point)
                    candidate[nearcand] = fastadd(candidate[nearcand], point)
                    center_index_cluster[i] = nearcand

    weightpoint=[[int(round(tmp/c[1])) for tmp in c[0]] for c in candidate if c[1]>=2]
    weightpoint=np.array(weightpoint)
    #clusternumber = weightpoint.shape
    print('Clustering Done')
    return weightpoint