# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 16:49:43 2017

@author: Niall Macquaide
"""

from __future__ import print_function
import sys
import argparse
import math
import pandas as pd
import numpy as np
from scipy import interpolate, signal, stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *
from os import path



def main():
    global args
    args = parse_command_line()

    (xdata, ydata) = read_csv_data(args.filename, 't (sec)', args.column_str)

    if not equally_spaced(xdata):
        error('x values not equally spaced')

    df = analyze_data(xdata, ydata)
    d1('transients analyzed: %d' % (len(df.index)))

    if args.stats_flag:
        df = df.describe()
    #    df.to_csv(sys.stdout, index=False)


def parse_command_line():
    parser = argparse.ArgumentParser(usage='%(prog)s [options] filename',
                                     version='Version 1.0')

    parser.add_argument('-s', '--stats',
                        action='store_true',
                        dest='stats_flag',
                        default=False,
                        help='output only summary statistics')

    parser.add_argument('-p', '--plot',
                        action='store_true',
                        dest='plot_flag',
                        default=False,
                        help='plot analysis')

    parser.add_argument('-d', '--debug',
                        type=int,
                        dest='debug_level',
                        default=0,
                        help='print debug output up to level n')

    parser.add_argument('-c', '--column',
                        type=str,
                        dest='column_str',
                        default='Vm',
                        help='select column for analysis (first string match')

    parser.add_argument('filename',
                        help='csv file containing simulation output')

    return parser.parse_args()


def error(msg):
    print('ERROR: ' + msg, file=sys.stderr)
    sys.exit()


def warn(*objs):
    print('WARNING:', *objs, file=sys.stderr)


def d1(*objs):
    print(*objs)


def d2(*objs):
    print(*objs)


def out(*objs):
    print(*objs, file=sys.stderr)


def read_csv_data(fn, xs, ys):
    df = pd.read_csv(fn)

    xcol = find_column_by_string(df.columns, xs, 'time', 0)
    ycol = find_column_by_string(df.columns, ys, 'parameter', 1)
    d1('x: ', xcol)
    d1('y: ', ycol)

    return (df[xcol].values, df[ycol].values)


def find_column_by_string(columns, s, cname, defaultcol):
    cols = [col for col in columns if s in col]
    if len(cols) == 0:
        xcol = columns[defaultcol]
        warn(cname + ' column string not found, using column ' + xcol)
    else:
        xcol = cols[0]
        if len(cols) > 1:
            warn(cname + ' column string not unique, using ' + xcol)
    return xcol


def equally_spaced(x):
    dx = np.diff(x)
    cdx = np.std(dx) / np.mean(dx)
    mcdx = 0.01
    d1('coefficient of variation of time step:', cdx)
    d1('maximum acceptable coefficient:', mcdx)
    return cdx < mcdx


def analyze_data(x, y):
    assert len(x) == len(y)
    if len(x) < 10:
        warn('at least 10 points needed for analysis')
        return 0

    (icross, ythresh) = find_threshold_crossings(x, y, 0.5)
    if len(icross) == 0:
        return 0

    (itrough, ipeak, istart, iend) = find_complete_transients(x, y, icross)
    if len(istart) == 0:
        return


    df = pd.DataFrame(columns=get_column_names())
    #   df = pd.DataFrame()
    #print(pdlist)
    
    #pdlist = []
    for i in list(range(len(istart))):
        xt = x[itrough[i]:iend[i] + 1].copy()
        yt = y[itrough[i]:iend[i] + 1].copy()
        xs = x[istart[i]]
        ys = y[istart[i]]
        pdict = analyze_transient(i + 1, xt, yt, xs, ys)
        print(pdict)
        df=df.concat(df, pdict)
        #pdlist.append(pdict)
        

    
    #df.append(pdict, ignore_index=True)
    print(df)
    return df


def get_column_names():
    return ['T0', 'Interval', 'AmpUp', 'AmpDn', 'TPk', \
            'Up90', 'Dn90', \
            'CD10', 'CD25', 'CD50', 'CD75', 'CD90', \
            'TU10', 'TU20', 'TU25', 'TU30', 'TU40', 'TU50', \
            'TU60', 'TU70', 'TU75', 'TU80', 'TU90', \
            'TD10', 'TD20', 'TD25', 'TD30', 'TD40', 'TD50', \
            'TD60', 'TD70', 'TD75', 'TD80', 'TD90']


def plot_whole_trace(x, y, eflag, ythresh, icross, itrough, ipeak, istart, iend):
    plt.figure(figsize=(15, 3))
    plt.plot(x, y)
    if eflag:
        plt.axhline(y=ythresh, color='r')
        plt.plot(x[icross], y[icross], 'r+')
        plt.plot(x[istart], y[istart], 'g*')
        plt.plot(x[istart], y[istart], 'g')
        plt.plot(x[itrough], y[itrough], 'b+')
        plt.plot(x[ipeak], y[ipeak], 'b+')
        for i, j in zip(itrough, ipeak):
            plt.plot([x[i], x[j]], [y[i], y[j]], color='black')
        plt.plot(x[iend], y[iend], 'b+')
    expand_yaxis()
    plt.show()


def expand_yaxis():
    (x1, x2, y1, y2) = plt.axis()
    e = 0.02 * (y2 - y1)
    plt.axis((x1, x2, y1 - e, y2 + e))


def find_threshold_crossings(x, y, levelfraction):
    ymin = np.amin(y)
    ymax = np.amax(y)
    ythresh = ymin + levelfraction * (ymax - ymin)
    d1('ymin: ', ymin)
    d1('ymax: ', ymax)
    d1('ythresh: ', ythresh)
    icross = []
    for i in list(range(len(y) - 1)):
        if (y[i] <= ythresh) and (y[i + 1] > ythresh):
            icross.append(i)
    if len(icross) == 0:
        warn('no threshold crossings found')
    else:
        d2('crossings: ', len(icross))
    return (icross, ythresh)


def find_complete_transients(x, y, ic):
    itrough = []
    ipeak = []
    istart = []
    iend = []
    nth = 0
    for i in ic:
        nth = nth + 1
        j1 = search_minimum(y, i, -1)
        j2 = search_maximum(y, i, +1)
        j3 = search_minimum(y, j2, +1)
        if (j1 is not None) and (j2 is not None) and (j3 is not None):
            # Find shoulder/bend at start of upstroke as maximum perpendicular
            # distance from trace points to line joining trace trough to peak
            x1 = x[j1]
            x2 = x[j2]
            y1 = y[j1]
            y2 = y[j2]
            # d2('nth=', nth, ' j1=',j1, ' y1=',y1, ' j2=',j2, ' y2=',y2)
            dmax = 0
            jmax = None
            for j in list(range(j1, j2 + 1)):
                d = perpendicular_distance(x1, y1, x2, y2, x[j], y[j])
                if d > dmax:
                    dmax = d
                    jmax = j
                # d2('d=', d, ' dmax=',dmax, ' jmax=',jmax)
            if jmax is not None:
                itrough.append(j1)
                ipeak.append(j2)
                istart.append(jmax)
                iend.append(j3)

    d1('complete transients found: %d' % (len(istart)))
    return (itrough, ipeak, istart, iend)


def search_minimum(y, istart, idir):
    n = len(y)
    i = istart
    while (i > 1) and (i < n - 1) and (y[i + idir] <= y[i]):
        i = i + idir
    if (i == 1) or (i == n - 1):
        return None
    else:
        return i


def search_maximum(y, istart, idir):
    n = len(y)
    i = istart
    while (i > 1) and (i < n - 1) and (y[i + idir] >= y[i]):
        i = i + idir
    if (i == 0) or (i == n - 1):
        return None
    else:
        return i


def perpendicular_distance(x1, y1, x2, y2, x0, y0):
    assert (x1 != x2) and (y1 != y2)
    return ((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) \
           / math.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))


def analyze_transient(nth, x, y, x0, y0):
    # tryout_second_derivative(nth, x, y, xs, ys)
    ymin = np.amin(y)
    d = {}
    d['T0'] = x0
    (xpeak, ypeak) = find_peak(x, y)
    d['AmpUp'] = ypeak - y0
    d['AmpDn'] = ypeak - ymin
    d['TPk'] = xpeak
       
    
    # find upstroke times
    u10 = find_percent_level_crossing(x, y, y0, ypeak, 10, 1)
    u20 = find_percent_level_crossing(x, y, y0, ypeak, 20, 1)
    u25 = find_percent_level_crossing(x, y, y0, ypeak, 25, 1)
    u30 = find_percent_level_crossing(x, y, y0, ypeak, 30, 1)
    u40 = find_percent_level_crossing(x, y, y0, ypeak, 40, 1)
    u50 = find_percent_level_crossing(x, y, y0, ypeak, 50, 1)
    u60 = find_percent_level_crossing(x, y, y0, ypeak, 60, 1)
    u70 = find_percent_level_crossing(x, y, y0, ypeak, 70, 1)
    u75 = find_percent_level_crossing(x, y, y0, ypeak, 75, 1)
    u80 = find_percent_level_crossing(x, y, y0, ypeak, 80, 1)
    u90 = find_percent_level_crossing(x, y, y0, ypeak, 90, 1)
    # find downstroke times
    d90 = find_percent_level_crossing(x, y, ymin, ypeak, 10, -1)
    d80 = find_percent_level_crossing(x, y, ymin, ypeak, 20, -1)
    d75 = find_percent_level_crossing(x, y, ymin, ypeak, 25, -1)
    d70 = find_percent_level_crossing(x, y, ymin, ypeak, 30, -1)
    d60 = find_percent_level_crossing(x, y, ymin, ypeak, 40, -1)
    d50 = find_percent_level_crossing(x, y, ymin, ypeak, 50, -1)
    d40 = find_percent_level_crossing(x, y, ymin, ypeak, 60, -1)
    d30 = find_percent_level_crossing(x, y, ymin, ypeak, 70, -1)
    d25 = find_percent_level_crossing(x, y, ymin, ypeak, 75, -1)
    d20 = find_percent_level_crossing(x, y, ymin, ypeak, 80, -1)
    d10 = find_percent_level_crossing(x, y, ymin, ypeak, 90, -1)
    #
    #fit exponential to t50-t90
    
    
    
    # save parameters to dicttionary
    d['Up90'] = u90 - u10
    d['Dn90'] = d90 - d10
    d['CD10'] = d90 - u10
    d['CD25'] = d75 - u25
    d['CD50'] = d50 - u50
    d['CD75'] = d25 - u75
    d['CD90'] = d10 - u90
    d['TU10'] = u10
    d['TU20'] = u20
    d['TU25'] = u25
    d['TU30'] = u30
    d['TU40'] = u40
    d['TU50'] = u50
    d['TU60'] = u60
    d['TU70'] = u70
    d['TU75'] = u75
    d['TU80'] = u80
    d['TU90'] = u90
    d['TD10'] = d10
    d['TD20'] = d20
    d['TD25'] = d25
    d['TD30'] = d30
    d['TD40'] = d40
    d['TD50'] = d50
    d['TD60'] = d60
    d['TD70'] = d70
    d['TD75'] = d75
    d['TD80'] = d80
    d['TD90'] = d90
    d['Interval'] = 1000
    
    return d


def plot_transient(nth, xt, yt, x0, y0, d, title,fn=None):
    ymin = np.amin(yt)
    ymax = np.amax(yt)

    fig=plt.figure()
    plt.plot(xt, yt, 'g')
    plt.plot([xt[0], x0], [y0, y0], 'grey')
    plt.axhline(y=ymin, color='grey')
    plt.axhline(y=ymax, color='grey')
    xm = [];
    ym = []
    xm.append(d['TPk']);
    ym.append(ymax)
    plt.plot(xm, ym, 'k+')
    xm = [];
    ym = []
    xm.append(d['TU10']);
    ym.append(y0 + (ymax - y0) * 0.1)
    xm.append(d['TU20']);
    ym.append(y0 + (ymax - y0) * 0.2)
    xm.append(d['TU25']);
    ym.append(y0 + (ymax - y0) * 0.25)
    xm.append(d['TU30']);
    ym.append(y0 + (ymax - y0) * 0.3)
    xm.append(d['TU40']);
    ym.append(y0 + (ymax - y0) * 0.4)
    xm.append(d['TU50']);
    ym.append(y0 + (ymax - y0) * 0.5)
    xm.append(d['TU60']);
    ym.append(y0 + (ymax - y0) * 0.6)
    xm.append(d['TU70']);
    ym.append(y0 + (ymax - y0) * 0.7)
    xm.append(d['TU75']);
    ym.append(y0 + (ymax - y0) * 0.75)
    xm.append(d['TU80']);
    ym.append(y0 + (ymax - y0) * 0.8)
    xm.append(d['TU90']);
    ym.append(y0 + (ymax - y0) * 0.9)
    plt.plot(xm, ym, 'r+')
    xm = [];
    ym = []
    xm.append(d['TD10']);
    ym.append(ymin + (ymax - ymin) * 0.9)
    xm.append(d['TD20']);
    ym.append(ymin + (ymax - ymin) * 0.8)
    xm.append(d['TD25']);
    ym.append(ymin + (ymax - ymin) * 0.75)
    xm.append(d['TD30']);
    ym.append(ymin + (ymax - ymin) * 0.7)
    xm.append(d['TD40']);
    ym.append(ymin + (ymax - ymin) * 0.6)
    xm.append(d['TD50']);
    ym.append(ymin + (ymax - ymin) * 0.5)
    xm.append(d['TD60']);
    ym.append(ymin + (ymax - ymin) * 0.4)
    xm.append(d['TD70']);
    ym.append(ymin + (ymax - ymin) * 0.3)
    xm.append(d['TD75']);
    ym.append(ymin + (ymax - ymin) * 0.25)
    xm.append(d['TD80']);
    ym.append(ymin + (ymax - ymin) * 0.2)
    xm.append(d['TD90']);
    ym.append(ymin + (ymax - ymin) * 0.1)
    plt.plot(xm, ym, 'b+')
    
    plt.title(title)
    expand_yaxis()
    plt.show()
    if fn is None:
        fig.savefig('c:/tmp/test0.png')
    else:
        fig.savefig(fn)
    plt.close(fig)
    

def calc_transient(nth, xt, yt, x0, y0, d):
    ymin = np.amin(yt)
    ymax = np.amax(yt)

    
    xm = [];
    ym = []
    xm.append(d['TPk']);
    ym.append(ymax)
    
    xm = [];
    ym = []
    xm.append(d['TU10']);
    ym.append(y0 + (ymax - y0) * 0.1)
    xm.append(d['TU20']);
    ym.append(y0 + (ymax - y0) * 0.2)
    xm.append(d['TU25']);
    ym.append(y0 + (ymax - y0) * 0.25)
    xm.append(d['TU30']);
    ym.append(y0 + (ymax - y0) * 0.3)
    xm.append(d['TU40']);
    ym.append(y0 + (ymax - y0) * 0.4)
    xm.append(d['TU50']);
    ym.append(y0 + (ymax - y0) * 0.5)
    xm.append(d['TU60']);
    ym.append(y0 + (ymax - y0) * 0.6)
    xm.append(d['TU70']);
    ym.append(y0 + (ymax - y0) * 0.7)
    xm.append(d['TU75']);
    ym.append(y0 + (ymax - y0) * 0.75)
    xm.append(d['TU80']);
    ym.append(y0 + (ymax - y0) * 0.8)
    xm.append(d['TU90']);
    ym.append(y0 + (ymax - y0) * 0.9)
    
    xm1,ym1=xm,ym
    xm = [];
    ym = []
    xm.append(d['TD10']);
    ym.append(ymin + (ymax - ymin) * 0.9)
    xm.append(d['TD20']);
    ym.append(ymin + (ymax - ymin) * 0.8)
    xm.append(d['TD25']);
    ym.append(ymin + (ymax - ymin) * 0.75)
    xm.append(d['TD30']);
    ym.append(ymin + (ymax - ymin) * 0.7)
    xm.append(d['TD40']);
    ym.append(ymin + (ymax - ymin) * 0.6)
    xm.append(d['TD50']);
    ym.append(ymin + (ymax - ymin) * 0.5)
    xm.append(d['TD60']);
    ym.append(ymin + (ymax - ymin) * 0.4)
    xm.append(d['TD70']);
    ym.append(ymin + (ymax - ymin) * 0.3)
    xm.append(d['TD75']);
    ym.append(ymin + (ymax - ymin) * 0.25)
    xm.append(d['TD80']);
    ym.append(ymin + (ymax - ymin) * 0.2)
    xm.append(d['TD90']);
    ym.append(ymin + (ymax - ymin) * 0.1)
    
    return(xm1,ym1,xm,ym)



def tryout_second_derivative(nth, xt, yt, xs, ys):
    f = interpolate.interp1d(xt, yt, kind='cubic')
    xi = np.linspace(xt[0], xt[-1], 1000)
    yi = f(xi)
    ymin = np.amin(yt)
    ymax = np.amax(yt)
    dyi = np.gradient(yi)
    dyi = normalize_data(dyi, ymin, ymax)
    d2yi = np.gradient(dyi)
    d2yi = normalize_data(d2yi, ymin, ymax)
    (xp, yp) = find_peak(xi, d2yi)
    if nth == 1:
        plt.figure()
        plt.plot(xt, yt)
        plt.plot([xt[0], xs], [ys, ys], 'grey')
        plt.axhline(y=ymin, color='grey')
        plt.axhline(y=ymax, color='grey')
        plt.plot(xi, dyi, 'c')
        plt.plot(xi, d2yi, 'r')
        plt.axvline(x=xp, color='r')
        expand_yaxis()
        plt.show()


def normalize_data(y, zmin, zmax):
    ymin = np.amin(y)
    ymax = np.amax(y)
    ys = (y.copy() - ymin) / (ymax - ymin)  # 0..1
    ys = ys * (zmax - zmin) + zmin
    return ys


def find_peak(x, y):
    ymax = np.amax(y)
    imax = np.nonzero(y == ymax)
    # extra [0] is to unpack tuple (bloody Python!)
    return (x[imax[0][0]], y[imax[0][0]])


def find_percent_level_crossing(x, y, ymin, ymax, pc, kdir):
    if (pc < 1) or (pc > 99):
        error('level crossing percent must be 1-99')
    level = ymin + (ymax - ymin) * (pc / 100.0000000)
    found = False
    if kdir == 1:  # search forwards from start
        i = 0
        while (i < len(y)) and not found:
            if y[i] > level:
                if y[i - 1] == y[i]:
                    xcross = 0
                else:
                    xcross = x[i] - (y[i] - level) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
                found = True
            i = i + 1;
    else:  # search backwards from end
        i = len(y) - 2
        while (i > 0) and not found:
            if y[i] > level:
                if y[i + 1] == y[i]:
                    xcross = 0
                else:
                    xcross = x[i] - (level - y[i]) * (x[i + 1] - x[i]) / (y[i] - y[i + 1])
                found = True
            i = i - 1;
    if not found:
        error('level crossing failed')
    else:
        return xcross


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[window_len // 2 - 1:-window_len // 2]


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def BaselineRefit(y):
    fit = getbaseline(y)
    # fit=slope*x+intercept
    corrected = y / (fit / fit.max())
    return corrected


def BaselineRefitsub(y, percentleft=0.1):
    fit = getbaseline(y, percentleft)
    # fit=slope*x+intercept
    corrected = y - fit
    return corrected


def getbaseline(y, percentleft=0.1):
    # very like method baseline refit in ImageJ by Francis Burton
    popt = None
    xtemp = np.float32(np.arange(0, y.size))
    ytemp = y.copy()
    ytemp = smooth(y, window_len=5)

    # print float(xtemp.size)/float(x[start:stop].size)
    # plot(xtemp,ytemp)
    while float(ytemp.size) / float(y.size) > percentleft:
        try:
            popt, pcov = curve_fit(func, np.float32(xtemp), np.float32(ytemp), p0=[ytemp[0], 1 / xtemp[-1], ytemp[-1]])

            fit = func(xtemp, *popt)
        except:
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.float32(xtemp), np.float32(ytemp))
            fit = slope * xtemp + intercept
        # plot(xtemp,fit)
        # show()
        xtemp = xtemp[ytemp < fit]
        ytemp = ytemp[ytemp < fit]

        # print float(xtemp.size)/float(x[start:stop].size)
        # print popt
    if popt is None:
        return (slope * np.arange(0, y.size) + intercept)
    else:
        return func(np.arange(0, y.size), *popt)


def getfn(FILEOPENOPTIONS):
    root = Tk()
    root.withdraw()
    fn = filedialog.askopenfilename(**FILEOPENOPTIONS)
    return fn


def getAPD90(data, interval, istart, istop,tiffn,savgolf1st=13,savgolf2nd=3,Smooth=False,translist=False):
    apdlist = []
    data = data[istart:istop]
    data1 = data.copy()
    x, y = (np.arange(data.size) * interval), (signal.savgol_filter(data, 191, 2))
    y = BaselineRefit(y)
    data = BaselineRefit(data)
    (icross, ythresh) = find_threshold_crossings(x, y, 0.6)
    #x, y = (np.arange(data.size) * interval), (signal.savgol_filter(data, savgolf1st, 2))
    x, y = (np.arange(data.size) * interval, data)
    y = BaselineRefit(y)

    # (itrough, ipeak, istart, iend) = find_complete_transients(x, y, icross)
    #average the transients
    a = []
    
    
    if len(icross) > 1:
        ipre = np.int64(np.round(50.0 / interval / 1000))
        ipost = np.int64(np.rint(np.diff(np.array(icross)).mean()) - ipre)
        for i in range(len(icross)):
            if len(data[icross[i] - ipre:icross[i] + ipost]) > 0:
                # a.append(data[:,1][icross[i]-ipre:icross[i]+ipost])
                if Smooth==False:
                  
                    a.append(data[icross[i] - ipre:icross[i] + ipost])
                else:			
                    a.append(signal.savgol_filter(data, savgolf2nd, 2)[icross[i] - ipre:icross[i] + ipost])		
                                       
            else:
                a.append(signal.savgol_filter(data, savgolf2nd, 2)[icross[i] - ipre:icross[i] + ipost])
                
        aa= np.array([len(aa) for aa in a])
        a=[bb[:aa.min()] for bb in a]
    
        avap = np.array(a).mean(axis=0)[:]
    else:
        ipre = np.int64(np.round(50.0 / interval / 1000))
        #avap = signal.savgol_filter(data, savgolf2nd, 2)[icross[0] - ipre:istop]
        avap = signal.savgol_filter(data, savgolf2nd, 2)[icross[0] - ipre:istop]
        
    win = signal.boxcar(5)
    #filtered = (signal.convolve(np.pad(avap, 10, 'reflect'), win, mode='same') / sum(win))[10:-10]
    filtered=avap.copy()
    pdlist = []
    # for i in xrange(len(istart)):
    # xt = x[itrough[i]-10:iend[i]+120].copy()
    # yt = y[itrough[i]-10:iend[i]+120].copy()
    # xs = x[istart[i]]
    # ys = y[istart[i]]
    xt, yt, xs, ys = np.arange(filtered.size) * interval, filtered, 0, avap[:10].mean()
    pdict = analyze_transient(0, xt, yt, xs, ys)
    pd.DataFrame([pdict]).to_csv(tiffn.lower().replace('.tif','apresults.csv'))
    # pdlist.append(pdict)

    def analysetranslist (a,Trans):
        xt, yt, xs, ys = np.arange(a.size) * interval, filtered, 0, a[:10].mean()
        pdict = analyze_transient(0, xt, yt, xs, ys)
        return pdict['TD80'] - pdict['TU50']
    
    #APD80list = [analysetranslist(aa) for aa in a]
    apd80 = pdict['TD80'] - pdict['TU50']
    #print APD80list
    title = (('APD 80 = %.3g s' % (apd80)))
    xm1,ym1,xm,ym=plot_transient(0, xt, yt, xs, ys, pdict, title,fn=tiffn.lower().replace('.tif','apd80fig.png'))

    print('APD 80 = %.3g s' % (apd80))
    # plt.title(('APD 80 = %.3g s' % (apd80)))
    # apdlist.append (apd80)
    # print('Mean APD 80 = %.3g s' % np.array(apdlist)[:3].mean())
    # np.savetxt(fn.lower().replace('.csv','apd80.txt'),np.array(apdlist)[:3].mean())
    np.savetxt(tiffn.lower().replace('.tif','apd80.txt'),np.array([apd80]))
    np.savetxt(tiffn.lower().replace('.tif', 'volt.txt'), np.array(data))
    #plt.savefig(tiffn.lower().replace('.tif','apd80fig.png'))
    fig = plt.figure()
    plt.plot(y)
    plt.scatter(icross, [ythresh] * len(icross))
    plt.show()
    return apd80
    #if translist==False:
    #   return apd80
    #else:
    #   return apd80,translist
        
    '''
    from glob import glob
    flist=glob('D:/data/Niall/mi/02052019'+'/*/*apd80.txt')
    a=np.array([np.ravel(np.loadtxt(fn))[0] for fn in flist])
    '''
