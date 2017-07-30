# -*- coding: utf-8 -*-
#####################################
#Author :Fenghaozhe ZhejiangUniversity,School of Statistics
#The last time for modification: 2017.5.10
#####################################
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

#################################################################
#Here we use some function to get the picture
#We use the truth that the file .mhd has already turn the pixel value into HU,then we can skip
#the step get HU to resample pictures
#Here we already have :
#1. a table df_node ,the "file" value store all the candidate image path
#Here we need some function to :
#1. resample the whole image
#2. Segment the lung part
#################################################################

def readimage(img_path):
    itk_img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(itk_img)  # indexs are z y x,the axis is -> x,\v y (0,0) is the left top
    shape=img_array.shape
    return img_array

def resample(img_path,new_spacing=[1,0.5,0.5]):
    itk_img=sitk.ReadImage(img_path)
    img_array=sitk.GetArrayFromImage(itk_img) #indexs are z y x,the axis is -> x,\v y (0,0) is the left top
    old_spacing=np.array(itk_img.GetSpacing())#Here the order is x,y,z ,so we must change the order
    old_spacing=np.array([old_spacing[2],old_spacing[1],old_spacing[0]])#the order is z,y,x
    resize_factor = old_spacing / new_spacing#Noticeing that the order is z,y,x,
    shape=img_array.shape
    new_real_shape = shape * resize_factor # type is array
    new_shape=np.round(new_real_shape) # Integer
    real_resize_factor = new_shape / shape # The order is z,y,x
    resimage= scipy.ndimage.interpolation.zoom(img_array, real_resize_factor, mode='nearest')
    shape=resimage.shape
    return resimage,shape

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    # val shows that this labeling has 23 different labels means 23 different connected region
    # counts shows howmany points does this region have
    #val=0 is air

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]#return the lung label
    else:
        return None


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


##Actually we don't need to do a real lung segment

def segment_lung_slice(ima,fill_lung_structures=True):
    #make through the ima
    for yt in range(ima.shape[0]):
        ima[yt,0] = ima[0,0]
    for xt in range(ima.shape[1]):
        ima[-1,xt] = ima[0,0]

    shape=ima.shape
    binary_image = np.array(ima > -320, dtype=np.int8) + 1
    #calculate the connected region
    labels = measure.label(binary_image)
    #fill the air around the person
    background_label = labels[0,0]
    # Fill the air around the person

    #一开始相当于按照>-320与<-320进行了区分,这里把外部空气区域也认为是2了，这里人身体外就都是纯白了
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (比形态选择要好)

    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        labeling = measure.label(binary_image-1)
        l_max = largest_label_volume(labeling, bg=0)
        if l_max is not None:  # This slice contains some other things (needn't be lung)
            binary_image[labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1,background are 0

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None:  # if This image has lungs
        binary_image[labels != l_max] = 0
    #Here we have make no lung part all zero
    #Lung with inner sth 1 ,other 0

    area = float(shape[0] * shape[1])
    val, count = np.unique(binary_image, return_counts=True)
    if count[val == 0 ] / area < 0.97:  # which means the area of lung in the picture is over 2%
        flag = 1
    else:
        flag = 0

    ##find some special points
    #######################################
    #for being careful ,I notice many details here,to avoid many other things

    segimage=deepcopy(ima)
    segimage[binary_image==0]=1024 # This step turned the non-lung position in this image to 0，and save the lung position
    return segimage,binary_image,flag

def segment_lung_mask(ima,fill_lung_structures=True):
    # binary_image:2 presents  image>-320,1presents image<-320
    #ima=split(ima)
    ima=through(ima)
    shape=ima.shape
    flag=[0 for i in range(shape[0])]
    flag=np.array(flag)
    binary_image = np.array(ima > -320, dtype=np.int8) + 1
    #calculate the connected region
    labels = measure.label(binary_image)
    #fill the air around the person
    background_label = labels[0,0,0]
    # Fill the air around the person

    #一开始相当于按照>-320与<-320进行了区分,这里把外部空气区域也认为是2了，这里人身体外就都是纯白了
    binary_image[background_label == labels] = 2
    '''
    for zt in range(binary_image.shape[0]):
	for xt in range(binary_image.shape[2]):
	    yt = binary_image.shape[1] - 1
	    while binary_image[zt, yt, xt] != 2:
		binary_image[zt, yt, xt] = 2
		yt -= 1
    '''
    # Method of filling the lung structures (比形态选择要好)

    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            ##eunmerate the whole binary image

            axial_slice = axial_slice - 1

            #to make label =0 the lung

            #label the connected region
            labeling = measure.label(axial_slice)

            #find the max label which is the segment area
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some other things (needn't be lung)
                binary_image[i][labeling != l_max] = 1
                #maybe this has some small problem

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1,background are 0

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)

    if l_max is not None:  # if This image has lungs
        binary_image[labels != l_max] = 0
    #Here we have make no lung part all zero
    #Lung with inner sth 1 ,other 0

    ##find some special points
    #######################################
    #for being careful ,I notice many details here,to avoid many other things
    
    area = float(shape[1] * shape[2])
    for i in range(shape[0]):
        val, count = np.unique(binary_image[i], return_counts=True)
        if count[val == 0 ] / area < 0.97:  # which means the area of lung in the picture is over 2%
            flag[i] = 1
     #######################################
    ##We want to add some sentence to judge if we successfully seg the image
    ##The best is to see that if lung is in shape[0]/2 slice
    ##Then we should test the flag to check if we successful seg the lung
    testsum=np.sum(np.array(flag==1))
    ##If we successfully seg the lung,then test part should all be 0
    testlen=len(flag)
    if float(testsum)/float(testlen)<0.1:
        #segment the image by each slice
        segimage = np.zeros(ima.shape, dtype=int)
        binary_image = np.zeros(ima.shape, dtype=int)
        for zt in range(ima.shape[0]):
            #zt = 79
            slice_image = ima[zt]
            sliceseg, slicemask, flag[zt] = segment_lung_slice(slice_image)
            segimage[zt] = sliceseg
            binary_image[zt] = slicemask
        testsum=np.sum(np.array(flag==1))
        testlen=len(flag)
        if float(testsum)/float(testlen)<0.1:
            return None, None, None
        else:
            return segimage, binary_image, flag

    segimage=deepcopy(ima)
    segimage[binary_image==0] = 1024 # This step turned the non-lung position in this image to 0，and save the lung position
    return segimage, binary_image, flag

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

'''
def seed_coord_cluster(index, clsize):
    numcoords = len(index[0])
    if numcoords == 0:
        return []
    coords = []
    for i in range(len(index[0])):
        coords.append([index[0][i], index[1][i], index[2][i]])

    clnum = 0
    index_cluster = 0 - np.ones(numcoords, dtype=int)
    steps = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1],
             [1, 0, 0], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0], [1, -1, 1], [1, -1, -1],
             [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, 1, 1], [-1, 1, -1], [-1, -1, 0], [-1, -1, 1],
             [-1, -1, -1]]
    # steps = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
    # clustering by seeds
    clusters = []
    for i in range(0, numcoords):
        if index_cluster[i] < 0:
            print("%d" % (i))
            clnum += 1
            cluster_stack = [i]
            index_cluster[i] = clnum - 1
            clusters.append([i])
            size = 1
            while len(cluster_stack) > 0 and size <= clsize:
                pind = cluster_stack.pop(0)
                for step in steps:
                    neighbor = [coords[i][0] + step[0], coords[i][1] + step[1], coords[i][2] + step[2]]
                    if coords.count(neighbor) > 0:
                        nind = coords.index(neighbor)
                        if index_cluster[nind] < 0:
                            size += 1
                            cluster_stack.append(nind)
                            index_cluster[nind] = clnum - 1
                            clusters[-1].append(nind)

    # calculate the cluster center
    clind = 0
    clend = False
    clcenters = []
    while index_cluster.count(clind) > 0:
        summary = [0.0, 0.0, 0.0]
        size = 0
        for i in range(len(index_cluster)):
            if index_cluster[i] == clind:
                size += 1
                summary = [summary[0] + coords[i][0], summary[1] + coords[i][1], summary[2] + coords[i][2]]
        center = np.array([round(summary[0]/size), round(summary[1]/size), round(summary[2]/size)], dtype=int)
        clcenters.append(center)

    #the coordination order is z, y, x
    return clcenters

def seed_volume_cluster(nodule_matrix, cluster_size=-1, eliminate_lower_size=5, result_vision=False):
    clnum = 0
    index_cluster = 0 - np.ones(nodule_matrix.shape, dtype=int)
    steps = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 1, -1], [0, -1, 0], [0, -1, 1], [0, -1, -1],
             [1, 0, 0], [1, 0, 1], [1, 0, -1], [1, 1, 0], [1, 1, 1], [1, 1, -1], [1, -1, 0], [1, -1, 1], [1, -1, -1],
             [-1, 0, 0], [-1, 0, 1], [-1, 0, -1], [-1, 1, 0], [-1, 1, 1], [-1, 1, -1], [-1, -1, 0], [-1, -1, 1],
             [-1, -1, -1]]
    # steps = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
    # clustering by seeds
    clusters = []
    for z in range(nodule_matrix.shape[0]):
        for y in range(nodule_matrix.shape[1]):
            for x in range(nodule_matrix.shape[2]):
                if nodule_matrix[z][y][x]>0 and index_cluster[z][y][x]<0:
                    clnum += 1
                    cluster_stack = [[z,y,x]]
                    index_cluster[z][y][x] = clnum - 1
                    clusters.append([[z,y,x]])
                    size = 1
                    while len(cluster_stack) > 0:
                        coord = cluster_stack.pop(0)
                        for step in steps:
                            neighbor = np.array([coord[0] + step[0], coord[1] + step[1], coord[2] + step[2]], dtype=int)
                            if not mt.coord_overflow(neighbor, nodule_matrix.shape) and nodule_matrix[neighbor[0],neighbor[1],neighbor[2]]>0 and index_cluster[neighbor[0],neighbor[1],neighbor[2]]<0:
                                size += 1
                                cluster_stack.append(neighbor)
                                index_cluster[neighbor[0],neighbor[1],neighbor[2]] = clnum - 1
                                clusters[-1].append(neighbor)
                        if cluster_size>0 and size>cluster_size:
                            break
    # calculate the cluster center
    clind = 0
    clcenters = []
    #cloutput = open("clsizes.txt", "w")
    for cluster in clusters:
        #cloutput.write("%d " %(len(cluster)))
        summary = np.array([0.0, 0.0, 0.0])
        for point in cluster:
            summary = [summary[0]+point[0], summary[1]+point[1], summary[2]+point[2]]
        if len(cluster)<=eliminate_lower_size:
            continue
        center = np.array([round(summary[0]/len(cluster)), round(summary[1]/len(cluster)), round(summary[2]/len(cluster))], dtype=int)
        #center = [int(round(center[0]/len(cluster))), int(round(center[1]/len(cluster))), int(round(center[2]/len(cluster)))]
        clcenters.append(center)
    #cloutput.close()
    
    if result_vision:
        rv = np.zeros(shape = (nodule_matrix.shape[0], nodule_matrix.shape[1], nodule_matrix.shape[2], 3))
        for cl in clusters:
            r = round(random.random(), 4)
            g = round(random.random(), 4)
            b = round(random.random(), 4)
            color = np.array([r, g, b])
            for coord in cl:
                rv[coord[0], coord[1], coord[2]] = color
        cv.view_CT(rv)

    #the coordination order is z, y, x
    return clcenters, index_cluster
    
def cluster_centers(cluster_labels, eliminate_upper_size=40000):
    num_labels = cluster_labels.max()
    cluster_sizes = np.zeros(shape=(num_labels), dtype=int)
    centers = np.zeros(shape=(num_labels, 3), dtype=float)
    for z in range(cluster_labels.shape[0]):
        for y in range(cluster_labels.shape[1]):
	    for x in range(cluster_labels.shape[2]):
	        label = cluster_labels[z][y][x]
		if cluster_sizes[label] >= 0:
		    cluster_sizes[label] += 1
		    centers[label] += np.array([z, y, x])
		    if cluster_sizes[label] > eliminate_upper_size:
		        #no longer to caluculate the center of this cluster for its too large
		        cluster_sizes[label] = -1
			
    clcenters = []
    for i in range(centers):
        if centers[i]>=0:
	    center = centers[i] / cluster_sizes[i]
            clcenters.append(center)

    return clcenters
'''