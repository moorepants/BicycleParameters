#!/usr/bin/env/ python

import os
from math import pi

import numpy as np
from numpy import ma
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from uncertainties import ufloat

# local modules
from io import load_pendulum_mat_file

def average_rectified_sections(data):
    '''Returns a slice of an oscillating data vector based on the max and min
    of the mean of the sections created by retifiying the data.

    Parameters
    ----------
    data : ndarray, shape(n,)

    Returns
    -------
    data : ndarray, shape(m,)
        A slice where m is typically less than n.

    Notes
    -----
    This is a function to try to handle the fact that some of the data from the
    torsional pendulum had a beating like phenomena and we only want to select
    a section of the data that doesn't seem to exhibit the phenomena.

    '''
    # subtract the mean so that there are zero crossings
    meanSubData = data - np.mean(data)
    # find the zero crossings
    zeroCrossings = np.where(np.diff(np.sign(meanSubData)))[0]
    # add a zero to the beginning
    crossings = np.concatenate((np.array([0]), zeroCrossings))
    # find the mean value of the rectified sections and the local indice
    secMean = []
    localMeanInd = []
    for sec in np.split(np.abs(meanSubData), zeroCrossings):
        localMeanInd.append(np.argmax(sec))
        secMean.append(np.mean(sec))
    meanInd = []
    # make the global indices
    for i, val in enumerate(crossings):
        meanInd.append(val + localMeanInd[i])
    # only take the top part of the data because some the zero crossings can be
    # a lot at one point mainly due to the resolution of the daq box
    threshold = np.mean(secMean)
    secMeanOverThresh = []
    indice = []
    for i, val in enumerate(secMean):
        if val > threshold:
            secMeanOverThresh.append(val)
            indice.append(meanInd[i])
    # now return the data based on the max value and the min value
    maxInd = indice[np.argmax(secMeanOverThresh)]
    minInd = indice[np.argmin(secMeanOverThresh)]

    return data[maxInd:minInd]

def calc_periods_for_files(directory, filenames, forkIsSplit):
    '''Calculates the period for all filenames in directory.

    Parameters
    ----------
    directory : string
        This is the path to the RawData directory.
    filenames : list
        List of all the mat file names in the RawData directory.
    forkIsSplit : boolean
        True if the fork is broken into a handlebar and fork and false if the
        fork and handlebar was measured together.

    Returns
    -------
    periods : dictionary
        Contains all the periods for the mat files in the RawData directory.

    '''

    periods = {}

    def pathParts(path):
        '''Splits a path into a list of its parts.'''
        components = []
        while True:
            (path,tail) = os.path.split(path)
            if tail == "":
                components.reverse()
                return components
            components.append(tail)

    pathToRawDataParts = pathParts(directory)
    pathToRawDataParts.pop()
    pathToBicycleDir = os.path.join(pathToRawDataParts[0],
                                    pathToRawDataParts[1],
                                    pathToRawDataParts[2])
    pathToPlotDir = os.path.join(pathToBicycleDir, 'Plots', 'PendulumFit')

    # make sure there is a place to save the plots
    if not os.path.exists(pathToPlotDir):
        os.makedirs(pathToPlotDir)

    for f in filenames:
        print "Calculating the period for:", f
        # load the pendulum data
        pathToMatFile = os.path.join(directory, f)
        matData = load_pendulum_mat_file(pathToMatFile)
        # generate a variable name for this period
        periodKey = get_period_key(matData, forkIsSplit)
        # calculate the period
        sampleRate = get_sample_rate(matData)
        pathToPlotFile = os.path.join(pathToPlotDir,
                                      os.path.splitext(f)[0] + '.png')
        period = get_period_from_truncated(matData['data'],
                                           sampleRate,
                                           pathToPlotFile)
        print "The period is:", period, "\n"
        # either append the the period or if it isn't there yet, then
        # make a new list
        try:
            periods[periodKey].append(period)
        except KeyError:
            periods[periodKey] = [period]

    # now average all the periods
    for k, v in periods.items():
        if k.startswith('T'):
            periods[k] = np.mean(v)

    return periods

def check_for_period(mp, forkIsSplit):
    '''Returns whether the fork is split into two pieces and whether the period
    calculations need to happen again.

    Parameters
    ----------
    mp : dictionary
        Dictionary the measured parameters.
    forkIsSplit : boolean
        True if the fork is broken into a handlebar and fork and false if the
        fork and handlebar was measured together.

    Returns
    -------
    forcePeriodCalc : boolean
        True if there wasn't enough period data in mp, false if there was.
    forkIsSplit : boolean
        True if the fork is broken into a handlebar and fork and false if the
        fork and handlebar was measured together.

    '''
    forcePeriodCalc = False
    #Check to see if mp contains at enough periods to not need
    # recalculation
    ncTSum = 0
    ntTSum = 0
    for key in mp.keys():
        # check for any periods in the keys
        if key[:2] == 'Tc':
            ncTSum += 1
        elif key[:2] == 'Tt':
            ntTSum += 1

    # if there isn't enough data then force the period cals again
    if forkIsSplit:
        if ncTSum < 5 or ntTSum < 11:
            forcePeriodCalc = True
    else:
        if ncTSum < 4 or ntTSum < 8:
            forcePeriodCalc = True

    return forcePeriodCalc

def fit_goodness(ym, yp):
    '''
    Calculate the goodness of fit.

    Parameters
    ----------
    ym : ndarray, shape(n,)
        The vector of measured values.
    yp : ndarry, shape(n,)
        The vector of predicted values.

    Returns
    -------
    rsq : float
        The r squared value of the fit.
    SSE : float
        The error sum of squares.
    SST : float
        The total sum of squares.
    SSR : float
        The regression sum of squares.

    '''
    SSR = sum((yp - np.mean(ym))**2)
    SST = sum((ym - np.mean(ym))**2)
    SSE = SST - SSR
    rsq = SSR / SST
    return rsq, SSE, SST, SSR

def get_period(data, sampleRate, pathToPlotFile):
    '''Returns the period and uncertainty for data resembling a decaying
    oscillation.

    Parameters
    ----------
    data : ndarray, shape(n,)
        A time series that resembles a decaying oscillation.
    sampleRate : int
        The frequency that data was sampled at.
    pathToPlotFile : string
        A path to the file to print the plots.

    Returns
    -------
    T : ufloat
        The period of oscillation and its uncertainty.

    '''

    y = data
    x = np.linspace(0., (len(y) - 1) / float(sampleRate), num=len(y))

    def fitfunc(p, t):
        '''Decaying oscillation function.'''
        a = p[0]
        b = np.exp(-p[3] * p[4] * t)
        c = p[1] * np.sin(p[4] * np.sqrt(1 - p[3]**2) * t)
        d = p[2] * np.cos(p[4] * np.sqrt(1 - p[3]**2) * t)
        return a + b * (c + d)

    # initial guesses
    #p0 = np.array([1.35, -.5, -.75, 0.01, 3.93]) # guess from delft
    #p0 = np.array([2.5, -.75, -.75, 0.001, 4.3]) # guess from ucd
    p0 = make_guess(data, sampleRate) # tries to make a good guess

    # create the error function
    errfunc = lambda p, t, y: fitfunc(p, t) - y

    # minimize the error function
    p1, success = leastsq(errfunc, p0[:], args=(x, y))

    lscurve = fitfunc(p1, x)

    # find the uncertainty in the fit parameters
    rsq, SSE, SST, SSR = fit_goodness(y, lscurve)
    sigma = np.sqrt(SSE / (len(y) - len(p0)))

    # calculate the jacobian
    L = jac_fitfunc(p1, x)

    # the Hessian
    H = np.dot(L.T, L)

    # the covariance matrix
    U = sigma**2. * np.linalg.inv(H)

    # the standard deviations
    sigp = np.sqrt(U.diagonal())

    # natural frequency
    wo = ufloat((p1[4], sigp[4]))
    # damping ratio
    zeta = ufloat((p1[3], sigp[3]))
    # damped natural frequency
    wd = (1. - zeta**2.)**(1. / 2.) * wo
    # damped natural frequency (hz)
    fd = wd / 2. / pi
    # period
    T = 1. / fd

    # plot the data and save it to file
    fig = plt.figure()
    plot_osfit(x, y, lscurve, p1, rsq, T, m=np.max(x), fig=fig)
    plt.savefig(pathToPlotFile)
    plt.close()

    # return the period
    return T

def get_period_from_truncated(data, sampleRate, pathToPlotFile):
    #dataRec = average_rectified_sections(data)
    dataRec = data
    dataGood = select_good_data(dataRec, 0.1)
    return get_period(dataGood, sampleRate, pathToPlotFile)

def get_period_key(matData, forkIsSplit):
    '''Returns a dictionary key for the period entries.

    Parameters
    ----------
    matData : dictionary
        The data imported from a pendulum mat file.
    forkIsSplit : boolean
        True if the fork is broken into a handlebar and fork and false if the
        fork and handlebar was measured together.

    Returns
    -------
    key : string
        A key of the form 'T[pendulum][part][orientation]'. For example, if it
        is the frame that was hung as a torsional pendulum at the second
        orientation angle then the key would be 'TtB2'.

    '''
    # set up the subscripting for the period key
    subscripts = {'Fwheel': 'F',
                  'Rwheel': 'R',
                  'Frame': 'B'}
    if forkIsSplit:
        subscripts['Fork'] = 'S'
        subscripts['Handlebar'] = 'G'
    else:
        subscripts['Fork'] = 'H'
    try:
        subscripts[matData['rod']] = 'P'
    except KeyError:
        subscripts['Rod'] = 'P'

    # used to convert word ordinals to numbers
    ordinal = {'First' : '1',
               'Second' : '2',
               'Third' : '3',
               'Fourth' : '4',
               'Fifth' : '5',
               'Sixth' : '6'}
    try:
        orienWord = matData['angleOrder']
    except:
        orienWord = matData['angle']
    pend = matData['pendulum'][0].lower()
    part = subscripts[matData['part']]
    orienNum = ordinal[orienWord]
    return 'T' + pend + part + orienNum

def get_sample_rate(matData):
    '''Returns the sample rate for the data.'''
    if 'ActualRate' in matData.keys():
        sampleRate = matData['ActualRate']
    else:
        sampleRate = matData['sampleRate']
    return sampleRate

def jac_fitfunc(p, t):
    '''
    Calculate the Jacobian of a decaying oscillation function.

    Uses the analytical formulations of the partial derivatives.

    Parameters
    ----------
    p : the five parameters of the equation
    t : time vector

    Returns
    -------
    jac : The jacobian, the partial of the vector function with respect to the
    parameters vector. A 5 x N matrix where N is the number of time steps.

    '''
    jac = np.zeros((len(p), len(t)))
    e = np.exp(-p[3] * p[4] * t)
    dampsq = np.sqrt(1 - p[3]**2)
    s = np.sin(dampsq * p[4] * t)
    c = np.cos(dampsq * p[4] * t)
    jac[0] = np.ones_like(t)
    jac[1] = e * s
    jac[2] = e * c
    jac[3] = (-p[4] * t * e * (p[1] * s + p[2] * c) + e * (-p[1] * p[3] * p[4]
              * t / dampsq * c + p[2] * p[3] * p[4] * t / dampsq * s))
    jac[4] = (-p[3] * t * e * (p[1] * s + p[2] * c) + e * dampsq * t * (p[1] *
              c - p[2] * s))
    return jac.T

def make_guess(data, sampleRate):
    '''Returns a decent starting point for fitting the decaying oscillation
    function.

    '''
    p = np.zeros(5)

    # the first unknown is the shift along the y axis
    p[0] = np.mean(data)

    # work with the mean subtracted data from now on
    data = data - p[0]

    # what is the initial slope of the curve
    if data[10] > data[0]:
        slope = 1
    else:
        slope = -1

    # the second is the amplitude for the sin function
    p[1] = slope * np.max(data) / 2

    # the third is the amplitude for the cos function
    p[2] = slope * np.max(data)

    # the fourth is the damping ratio and is typically small, 0.001 < zeta < 0.02
    p[3] = 0.001

    # the fifth is the undamped natural frequency
    # first remove the data around zero
    dataMasked = ma.masked_inside(data, -0.1, 0.1)
    # find the zero crossings
    zeroCrossings = np.where(np.diff(np.sign(dataMasked)))[0]
    # remove redundant crossings
    zero = []
    for i, v in enumerate(zeroCrossings):
        if abs(v - zeroCrossings[i - 1]) > 20:
            zero.append(v)
    # get the samples per period
    samplesPerPeriod = 2*np.mean(np.diff(zero))
    # now the frequency
    p[4] = (samplesPerPeriod / float(sampleRate) /2. / pi)**-1
    if np.isnan(p[4]):
        p[4] = 4.

    return p

def plot_osfit(t, ym, yf, p, rsq, T, m=None, fig=None):
    '''Plot fitted data over the measured

    Parameters
    ----------
    t : ndarray (n,)
        Measurement time in seconds
    ym : ndarray (n,)
        The measured voltage
    yf : ndarray (n,)
    p : ndarray (5,)
        The fit parameters for the decaying osicallation fucntion
    rsq : float
        The r squared value of y (the fit)
    T : float
        The period
    m : float
        The maximum value to plot

    Returns
    -------
    fig : the figure

    '''
    # figure properties
    figwidth = 8. # in inches
    goldenMean = (np.sqrt(5) - 1.0) / 2.0
    figsize = [figwidth, figwidth * goldenMean]
    params = {#'backend': 'ps',
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'text.fontsize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'text.usetex': True,
        #'figure.figsize': figsize
        }
    if fig:
        fig = fig
    else:
        fig = plt.figure(2)
    fig.set_size_inches(figsize)
    plt.rcParams.update(params)
    ax1 = plt.axes([0.125, 0.125, 0.9-0.125, 0.65])
    #if m == None:
        #end = len(t)
    #else:
        #end = t[round(m/t[-1]*len(t))]
    ax1.plot(t, ym, '.', markersize=2)
    plt.plot(t, yf, 'k-')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    equation = r'$f(t)={0:1.2f}+e^{{-({3:1.3f})({4:1.1f})t}}\left[{1:1.2f}\sin{{\sqrt{{1-{3:1.3f}^2}}{4:1.1f}t}}+{2:1.2f}\cos{{\sqrt{{1-{3:1.3f}^2}}{4:1.1f}t}}\right]$'.format(p[0], p[1], p[2], p[3], p[4])
    rsquare = '$r^2={0:1.3f}$'.format(rsq)
    period = '$T={0} s$'.format(T)
    plt.title(equation + '\n' + rsquare + ', ' + period)
    plt.legend(['Measured', 'Fit'])
    if m is not None:
        plt.xlim((0, m))
    else:
        pass
    return fig

def select_good_data(data, percent):

    '''Returns a slice of the data from the index at maximum value to the index
    at a percent of the maximum value.

    Parameters
    ----------
    data : ndarray, shape(1,)
        This should be a decaying function.
    percent : float
        The percent of the maximum to clip.

    This basically snips of the beginning and end of the data so that the super
    damped tails are gone and also any weirdness at the beginning.

    '''
    meanSub = data - np.mean(data)
    maxVal = np.max(np.abs(meanSub))
    maxInd = np.argmax(np.abs(meanSub))
    for i, v in reversed(list(enumerate(meanSub))):
        if v > percent * maxVal:
            minInd = i
            break

    return data[maxInd:minInd]
