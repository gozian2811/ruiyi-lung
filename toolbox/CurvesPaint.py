import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, BSpline, BarycentricInterpolator

def polynomial(x, p, deg):
    result = np.zeros_like(x)
    for i in range(deg):
        result += p[i]*x**(deg-i)
    return result

def curves_paint(curve_path = '.', curve_files = None, output_file = None, name_loc=None, xlabel=None, ylabel=None, xlim=-1):
    if curve_path is not None and curve_files is not None:
        for rf in range(len(curve_files)):
            curve_files[rf] = curve_path + '/' + curve_files[rf]
    elif curve_path is not None:
        curve_files = glob.glob(curve_path + "/*.npy")
    lines = []
    names = []
    for cf in range(len(curve_files)):
        curve_file = curve_files[cf]
        name = os.path.splitext(os.path.basename(curve_file))[0]
        curve = np.load(curve_file)
        indices = range(len(curve))
        line, = plt.plot(indices[:xlim], curve[:xlim], alpha=0.8)
        lines.append(line)
        names.append(name)
    plt.legend(lines, names, loc=name_loc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, format="pdf")
        plt.close()

def roc_paint(roc_path = '.', roc_files = None, filter = True, output_file = 'roc_curve.pdf'):
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
    #plt.show()
    plt.savefig(output_file, format="pdf")
    plt.close()

curves_paint(curve_path='../experiments/trains/loss_cl', output_file='../loss_curve_classification.pdf', curve_files=['train.npy', 'val.npy'], xlabel='epoch', ylabel='loss', xlim=100)
#roc_paint(roc_path='../experiments/rocs_trand', roc_files=['Ipris.npy', 'DenseNet.npy', 'MC-CNN.npy', 'DenseCropNet.npy'], filter=True, output_file='../roc_curve_sota.pdf')