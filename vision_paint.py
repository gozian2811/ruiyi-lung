import os
import glob
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d, BSpline, BarycentricInterpolator

def polynomial(x, p, deg):
    result = np.zeros_like(x)
    for i in range(deg):
        result += p[i]*x**(deg-i)
    return result

def curves_paint(curve_path='.', curve_files=None, output_file=None, xlabel=None, ylabel=None, names_loc='lower right', length_limit=0, smooth=0, colors=None, linestyles=None):
    if curve_path is not None and curve_files is not None:
        for rf in range(len(curve_files)):
            curve_files[rf] = curve_path + '/' + curve_files[rf]
    elif curve_path is not None:
        curve_files = glob.glob(curve_path + "/*.npy")[::-1]
    lines = []
    names = []
    #if len(curve_files)<=8: colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for cf in range(len(curve_files)):
        curve_file = curve_files[cf]
        name = os.path.splitext(os.path.basename(curve_file))[0]
        curve = np.load(curve_file)
        if curve.max()<1: curve *= 100
        if smooth>0:
            #curve_smooth = np.ones(len(curve)+2*smooth) * curve.mean()
            #curve_smooth[smooth:len(curve_smooth)-smooth] = curve
            curve_smooth = np.pad(curve, (smooth, smooth), 'constant', constant_values=(curve[:smooth].mean(), curve[-smooth:].mean()))
            for c in range(len(curve)):
                #step = min(min(c, len(curve)-c), smooth)
                #if step>0:
                #    curve_smooth[c] = curve[c-step:c+step].mean()
                curve[c] = curve_smooth[c:c+2*smooth].mean()
            #curve = curve_smooth
        if length_limit>0:
            curve = curve[:length_limit]
        indices = range(1, len(curve)+1)
        if colors is not None:
            line, = plt.plot(indices, curve, alpha=0.8, color=colors[cf])
        elif linestyles is not None:
            line, = plt.plot(indices, curve, alpha=0.8, linestyle=linestyles[cf])
        else:
            line, = plt.plot(indices, curve, alpha=0.8)
        plt.xlabel(xlabel, fontproperties = 'STSong')
        plt.ylabel(ylabel, fontproperties = 'STSong')
        plt.xticks(fontproperties = 'Times New Roman')
        plt.yticks(fontproperties = 'Times New Roman')
        lines.append(line)
        names.append(name)
    plt.legend(lines, names, loc=names_loc, prop={'family': 'Times New Roman'})
    plt.grid(True)
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, format="pdf")
        plt.close()

def roc_paint(roc_path = '.', roc_files = None, filter = True, output_file = None):
    if roc_path is not None and roc_files is not None:
        for rf in range(len(roc_files)):
            roc_files[rf] = roc_path + '/' + roc_files[rf]
    elif roc_path is not None:
        roc_files = glob.glob(roc_path+"/*.npy")
    lines = []
    names = []
    for rf in range(len(roc_files)):
        roc_file = roc_files[rf]
        name = os.path.splitext(os.path.basename(roc_file))[0]
        roc = np.load(roc_file)
        '''
        start = len(roc[0][roc[1]==roc[1][0]])
        TPs = [roc[0][0], roc[0][start-1]]
        FPs = [roc[1][0], roc[1][start-1]]
        for r in range(start, roc.shape[1]-1):
            if roc[0][r] != TPs[-1] and roc[1][r] != FPs[-1]:
                TPs.append(roc[0][r])
                FPs.append(roc[1][r])
        TPs.append(roc[0][-1])
        FPs.append(roc[1][-1])
        xrange = [i for i in range(len(FPs))]
        '''
        filtering = filter
        while filtering:
            filtering = False
            r = 1
            while r < roc.shape[1]-1:
                if roc[1][r]!=roc[1][r-1] and (roc[0][r]-roc[0][r-1])/(roc[1][r]-roc[1][r-1])<(roc[0][r+1]-roc[0][r])/(roc[1][r+1]-roc[1][r]):
                    roc = np.delete(roc, r, axis=1)
                    filtering = True
                else:
                    r += 1
        xstart = len(roc[0][roc[1] == roc[1][0]]) - 1
        xnew1 = np.linspace(roc[1][0], roc[1][-1], 80)
        xnew2 = np.linspace(roc[1][0], roc[1][-1], 20)
        ynew = np.linspace(roc[0][0], roc[0][-1], 40)
        #ysmooth = interp1d(roc[1][xstart:], roc[0][xstart:], 'cubic')(xnew)
        ysmooth1 = interp1d(roc[1][xstart:], roc[0][xstart:], 'linear')(xnew1)
        ysmooth2 = BSpline(xnew1, ysmooth1, 4)(xnew2)
        xsmooth = interp1d(roc[0], roc[1], 'linear')(ynew)
        xfusion = xsmooth.copy()
        yfusion = ynew.copy()
        xsidx = 0
        xi = 0
        #while xi<len(xfusion) and xsidx<len(xnew):
        #    if xfusion[xi]>=xnew[xsidx] and yfusion[xi]>=ysmooth[xsidx]:
        #        xfusion = np.insert(xfusion, xi, xnew[xsidx])
        #        yfusion = np.insert(yfusion, xi, ysmooth[xsidx])
        #        xsidx += 1
        #    xi += 1
        #xfusion = np.concatenate((xfusion, xnew[xsidx:]), axis=0)
        #yfusion = np.concatenate((yfusion, ysmooth[xsidx:]), axis=0)
        line, =plt.plot(roc[1], roc[0], alpha=0.8)
        #line, = plt.plot(xfusion, yfusion)
        #line, = plt.plot(xnew1, ysmooth1)
        #line, = plt.plot(xnew2, ysmooth2)
        #plt.plot(ynew, xsmooth)
        lines.append(line)
        names.append(name)
    plt.legend(lines, names, loc='lower right')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(False)
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, format="pdf")
        plt.close()

def scatter_paint_datasets(scatter_sources, filewise=False, names=None):
    pslist = []
    if names is None:
        names = []
        for sf in range(len(scatter_sources)):
            name = os.path.splitext(os.path.basename(scatter_sources[sf]))[0]
            names.append(name)
    if filewise:
        namestarts = []
        pointlist = []
        for sf in range(len(scatter_sources)):
            namestarts.append(len(pointlist))
            pointfiles = os.listdir(scatter_sources[sf])
            for pointfile in pointfiles:
                pointlist.append(np.load(scatter_sources[sf]+'/'+pointfile))
        namestarts.append(len(pointlist))
        pointset = np.array(pointlist)
        TSNE(n_components=2).fit(pointset)
        for sf in range(len(names)):
            pointsets = plt.scatter(pointset[namestarts[sf]:namestarts[sf+1], 0], pointset[namestarts[sf]:namestarts[sf+1], 1], s=1, alpha=0.7)
            pslist.append(pointsets)
        #plt.scatter(pointset[:, 0], pointset[:, 1], s=1)
    else:
        for sf in range(len(scatter_sources)):
            pointset = np.load(scatter_sources[sf])
            pointsets = plt.scatter(pointset[:, 0], pointset[:, 1], s=1)
            pslist.append(pointsets)
    plt.xticks(fontproperties = 'Times New Roman')
    plt.yticks(fontproperties = 'Times New Roman')
    plt.legend(pslist, names, loc='lower right', prop={'family': 'Times New Roman'})
    plt.grid(True)
    plt.show()

def scatter_paint(point_file, label_file):
    pslist = []
    names = []
    points = np.load(point_file)
    labels = np.load(label_file)
    colors = labels
    for p in range(len(points)):
        if points[p].argmax() != labels[p]:
            colors[p] += 2
    plt.scatter(points[:,0], points[:,1], s=1, c=colors)
    plt.show()

#scatter_paint('experiments/experiments_cl/RegularlySharedNet_classification_4_fold3_epoch982_val/RegularlySharedNet_classification_4_fold3_epoch982_scores2.npy',
#              'experiments/experiments_cl/RegularlySharedNet_classification_4_fold3_epoch982_val/RegularlySharedNet_classification_4_fold3_epoch982_labels2.npy')
#scatter_paint_datasets(['experiments/experiments_cl/BasicNet_lidc_classification_9_epoch335_train/BasicNet_lidc_classification_9_epoch335_scores.npy',
#                        'experiments/experiments_cl/BasicNet_lidc_classification_9_epoch335_train_sph/BasicNet_lidc_classification_9_epoch335_scores.npy'], names=['LIDC-IDRI', 'SPH'])
scatter_paint_datasets(["experiments/experiments_cl/MultiTaskNet_classification_3_fold1_epoch900/MultiTaskNet_classification_3_fold1_epoch900_feature_avgp",
                        "experiments/experiments_cl/MultiTaskNet_classification_3_fold1_epoch900/MultiTaskNet_classification_3_fold1_epoch900_feature2_avgp"],
                        filewise=True, names=['LIDC-IDRI', 'LA-SPH'])
#roc_paint(roc_path="experiments/rocs_trand", roc_files=["$L=4$ $k=12$.npy", "$L=4$ $k=14$.npy", "$L=4$ $k=16$.npy"], xlabel='假阳率', ylabel='真阳率', output_file="roc_curve L=4.pdf")
#roc_paint(roc_path="experiments/rocs_trand", roc_files=["$L=2$ $k=14$.npy", "$L=3$ $k=14$.npy", "$L=4$ $k=14$.npy", "$L=5$ $k=14$.npy"], xlabel='假阳率', ylabel='真阳率', output_file="roc_curve k=14.pdf")
#curves_paint(curve_path="experiments/validations/p111", names_loc='lower right', length_limit=0, smooth=11, linestyles=['dashed', 'dashdot'])
#curves_paint(curve_path="experiments/validations/baseweightsharedbnnet_lidc_bNA", xlabel='轮次', ylabel='准确率', length_limit=0, smooth=6)
#curves_paint(curve_path="experiments/validations/baseweightsharednet_densecropnet_lidc_bNA", xlabel='轮次', ylabel='准确率', names_loc='middle right', length_limit=0, smooth=6)
#curves_paint(curve_path="experiments/validations/baseweightsharednet_densecropnet_sph_bNA", xlabel='轮次', ylabel='准确率', names_loc='lower right', length_limit=0, smooth=12)
#roc_paint(roc_path='experiments/experiments_cl', roc_files=['BasicNet_sph_classification_roc.npy', 'BaseweightSharedNet_sph_classification_roc.npy', 'BaseweightSharedNet_sph_classification_b4_roc.npy'], filter=False, output_file=None)
#roc_paint(roc_path='experiments/experiments_cl', roc_files=['BasicNet_lidc_classification_roc.npy', 'BaseweightSharedNet_lidc_classification_roc.npy', 'BaseweightSharedNet_lidc_classification_b4_roc.npy'], filter=True, output_file=None)
