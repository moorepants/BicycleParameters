import os
import re
import pickle
from math import pi
import numpy as np
from numpy import ma
from numpy.linalg import inv
from scipy.optimize import leastsq, newton
from scipy.io import loadmat
import matplotlib.pyplot as plt
from uncertainties import ufloat, unumpy

class Bicycle(object):
    '''An object for a bicycle. A bicycle has parameters. That's about it for
    now.

    '''

    def __new__(cls, shortname, forceRawCalc=False):
        '''Returns a NoneType object if there is no directory'''
        # is there a data directory for this bicycle? if not, tell the user to
        # put some data in the folder so we have something to work with!
        try:
            if os.path.isdir(os.path.join('bicycles', shortname)) == True:
                print "We have foundeth a directory named: bicycles/" + shortname
                return super(Bicycle, cls).__new__(cls)
            else:
                raise ValueError
        except:
            a = "Are you nuts?! Make a directory with basic data for your "
            b = "bicycle in bicycles/shortname, where 'shortname' is the "
            c = "capitalized one word name of your bicycle. Then I can "
            d = "actually created a bicycle object."
            print a + b + c + d
            return None

    def __init__(self, shortname, forceRawCalc=False):
        '''
        Sets the parameters if there any that are already saved.

        Arguments
        ---------
        shortname : string
            shortname of your bicicleta, one word, first letter is capped and
            should match a directory under bicycles/

        forceRawCalc : boolean
            Force a recalculation of the parameters from the raw data, else it
            will only do this calculation if there are no parameter files.

        '''

        self.shortname = shortname
        self.directory = os.path.join('bicycles', shortname)
        self.parameters = {}

        # if you want to force a recalculation and there is a RawData directory
        if forceRawCalc and 'RawData' in os.listdir(self.directory):
            self.parameters['Benchmark'] = self.calculate_from_measured()
        elif not forceRawCalc and 'Parameters' not in os.listdir(self.directory):
            self.parameters['Benchmark'] = self.calculate_from_measured()
        elif not forceRawCalc and 'Parameters' in os.listdir(self.directory):
            parDir = os.path.join(self.directory, 'Parameters')
            parFiles = os.listdir(parDir)
            for parFile in parFiles:
                # remove the extension
                fname = os.path.splitext(parFile)[0]
                # get the bike and the parameter set type
                bike, ptype = space_out_camel_case(fname, output='list')
                # load the parameters
                pathToFile = os.path.join(parDir, parFile)
                self.parameters[ptype] = load_parameter_text_file(pathToFile)
        else:
            print "Where's the data?"

    def save(self, filetype='text'):
        '''
        Saves all the parameters to file.

        filetype : string
            'pickle' : python pickled dictionary
            'matlab' : matlab .mat file
            'text' : comma delimited text file

        '''

        if filetype == 'pickle':
            for k, v in self.params.items():
                thefile = self.directory + self.shortname + k + '.p'
                f = open(thefile, 'w')
                pickle.dump(v, f)
                f.close()
        elif filetype == 'matlab':
            # this should handle the uncertainties properly
            print "Doesn't work yet"

        elif filetype == 'text':
            print "Doesn't work yet"

    def calculate_from_measured(self, forcePeriodCalc=False):
        '''Calculates the parameters from measured data.

        '''
        rawDataDir = os.path.join(self.directory, 'RawData')
        pathToFile = os.path.join(rawDataDir, self.shortname + 'Measured.txt')

        # load the measured parameters
        mp = load_parameter_text_file(pathToFile)

        # if the the user doesn't specifiy to force period calculation, then
        # see if enough data is actually available in the Measured.txt file to
        # do the calculations
        if not forcePeriodCalc:
            # check to see if mp contains at enough periods to not need
            # recalculation
            ncTSum = 0
            ntTSum = 0
            isForkSplit = False
            for key in mp.keys():
                # check for any periods in the keys
                if key[:2] == 'Tc':
                    ncTSum += 1
                elif key[:2] == 'Tt':
                    ntTSum += 1
                # if there is an 'S' then the fork is split in two parts
                if key[:1] == 'S' or key[1:2] == 'S':
                    isForkSplit = True

            print "ncTSum:", ncTSum
            print "ntTSum:", ntTSum

            # if there isn't enough data then force the period cals again
            if isForkSplit:
                if ncTSum < 5 or ntTSum < 11:
                    forcePeriodCalc = True
            else:
                if ncTSum < 4 or ntTSum < 8:
                    forcePeriodCalc = True

        print "isForkSplit:", isForkSplit
        print "forcePeriodCalc", forcePeriodCalc

        if forcePeriodCalc == True:
            # get the list of mat files associated with this bike
            matFiles = [x for x in os.listdir(rawDataDir) if
                        x.startswith(self.shortname) and x.endswith('.mat')]
            # set up the subscripting for the period key
            subscripts = {'Fwheel': 'F',
                          'Rwheel': 'R',
                          'Frame': 'B'}
            if isForkSplit:
                subscripts['Fork'] = 'S'
                subscripts['Handlebar'] = 'H'
            else:
                subscripts['Fork'] = 'H'
            ordinal = {'First' : '1',
                       'Second' : '2',
                       'Third' : '3',
                       'Fourth' : '4',
                       'Fifth' : '5',
                       'Sixth' : '6'}
            # calculate the period for each file for this bicycle
            for f in matFiles:
                print "Calculating the period for:", f
                matData = load_pendulum_data_mat_file(os.path.join(rawDataDir,
                    f))
                if 'ActualRate' in matData.keys():
                    sampleRate = matData['ActualRate']
                else:
                    sampleRate = matData['sampleRate']
                try:
                    angle = matData['angleOrder']
                except:
                    angle = matData['angle']
                pend = matData['pendulum'][0].lower()
                part = subscripts[matData['part']]
                orien = ordinal[angle]
                key = 'T' + pend + part + orien
                period = get_period_from_truncated(matData['data'],
                        sampleRate)
                print key, period
                # either append the the period or if it isn't there yet, then
                # make a new list
                try:
                    mp[key].append(period)
                except KeyError:
                    mp[key] = [period]
            # now average all the periods
            for k, v in mp.items():
                if k.startswith('T'):
                    mp[k] = np.mean(v)

        print '\n'

        for k, v in mp.items():
            if k.startswith('T'):
                print k, v
        # calculate all the benchmark parameters
        par = {}

        # calculate the wheel radii
        par['rF'] = mp['dR'] / 2./ pi / mp['nF']
        par['rR'] = mp['dR'] / 2./ pi / mp['nR']

        # calculate the frame/fork fundamental geometry
        if ['w'] in mp.keys():
            # steer axis tilt in radians
            par['lambda'] = pi/180.*(90. - mp['headTubeAngle'])
            # calculate the front wheel trail
            forkOffset = mp['forkOffset']
            # wheelbase
            par['w'] = mp['w']
        else:
            a = mp['h1'] + mp['h2'] - mp['h3'] + .5 * mp['d1'] - .5 * mp['d2']
            b = mp['h4'] - .5 * mp['d3'] - mp['h5'] + .5 * mp['d4']
            c = np.sqrt(-(a - b)**2 + (mp['d'] + .5 * (mp['d2'] + mp['d3'])))
            par['lambda'] = lambda_from_abc(par['rF'], par['rR'], a, b, c)
            forkOffset = b
            par['w'] = (a + b) * np.cos(par['lambda']) + c * np.sin(par['lambda'])

        # trail
        par['c'] = trail(par['rF'], par['lambda'], forkOffset)[0]

        # calculate the frame rotation angle
        # alpha is the angle between the negative z pendulum (horizontal) and the
        # positive (up) steer axis, rotation about positive y
        alphaFrame = mp['frameAngle']
        # beta is the angle between the x bike frame and the x pendulum frame, rotation
        # about positive y
        betaFrame = par['lambda'] - alphaFrame * pi / 180

        # calculate the slope of the CoM line
        frameM = -np.tan(betaFrame)

        # calculate the z-intercept of the CoM line
        # frameMassDist is positive according to the pendulum ref frame
        frameMassDist = mp['frameMassDist']
        cb = unumpy.cos(betaFrame)
        frameB = -frameMassDist/cb - par['rR']

        # calculate the fork rotation angle
        betaFork = par['lambda'] - mp['forkAngle']*pi/180.

        # calculate the slope of the fork CoM line
        forkM = -unumpy.tan(betaFork)

        # calculate the z-intercept of the CoM line
        forkMassDist = mp['forkMassDist']
        cb = unumpy.cos(betaFork)
        tb = unumpy.tan(betaFork)
        forkB = - par['rF'] - forkMassDist/cb + par['w']*tb

        # intialize the matrices for the center of mass locations
        frameCoM = zeros((2), dtype='object')
        forkCoM = zeros((2), dtype='object')

        comb = np.array([[0, 1], [0, 2], [1, 2]])
        # calculate the frame center of mass position
        # initialize the matrix to store the line intersections
        lineX = zeros((3, 2), dtype='object')
        # for each line intersection...
        for j, row in enumerate(comb):
            a = unumpy.matrix(np.vstack([-frameM[row], np.ones((2))]).T)
            b = frameB[row]
            lineX[j] = np.dot(a.I, b)
        frameCoM[:] = np.mean(lineX, axis=0)
        # calculate the fork center of mass position
        # reinitialize the matrix to store the line intersections
        lineX = zeros((3, 2), dtype='object')
        # for each line intersection...
        for j, row in enumerate(comb):
            a = unumpy.matrix(np.vstack([-forkM[row], np.ones((2))]).T)
            b = forkB[row]
            lineX[j] = np.dot(a.I, b)
        forkCoM[:] = np.mean(lineX, axis=0)

        par['xB'] = frameCoM[0]
        par['zB'] = frameCoM[1]
        par['xH'] = forkCoM[0]
        par['zH'] = forkCoM[1]

        return par

def trail(rF, lam, fo):
    '''Caluculate the trail and mechanical trail

    Parameters:
    -----------
    rF: float
        The front wheel radius
    lam: float
        The steer axis tilt (pi/2 - headtube angle). The angle between the
        headtube and a vertical line.
    fo: float
        The fork offset

    Returns:
    --------
    c: float
        Trail
    cm: float
        Mechanical Trail

    '''

    # trail
    c = (rF * np.sin(lam) - fo) / np.cos(lam)
    # mechanical trail
    cm = c * np.cos(lam)
    return c, cm

def lambda_from_abc(rF, rR, a, b, c):
    '''Returns the steer axis tilt, lamba, for the parameter set based on the
    offsets from the steer axis.

    '''
    def lam_equality(lam, rF, rR, a, b, c):
        return np.sin(lam) - (rF - rR + c * np.cos(lam)) / (a + b)
    guess = np.arctan(c / (a + b)) # guess based on equal wheel radii
    
    # the following assumes that the uncertainty caluclated for the guess is
    # the same as the uncertainty for the true solution. This is not true! and
    # will surely breakdown the further the guess is away from the true
    # solution. There may be a way to calculate the correct uncertainity, but
    # that needs to be figured out.

    lam = newton(lam_equality, guess.nominal_value, args=(rF, rR, a, b, c))
    return ufloat((lam, guess.std_dev()))

def get_period_from_truncated(data, sampleFrequency):
    #dataRec = average_rectified_sections(data)
    dataRec = data
    #dataGood = select_good_data(dataRec, 0.1)
    dataGood = dataRec
    return get_period(dataGood, sampleFrequency)

def select_good_data(data, percent):
    '''Returns a slice of the data from the maximum value to a percent of the
    max.

    Parameters
    ----------
    data : ndarray, shape(1,)
        This should be a decaying function.
    percent : float
        The percent of the maximum to clip.

    This basically snips of the beginning and end of the data so that the super
    damped tails are gone and any weirdness at the beginning.

    '''
    maxVal = np.max(np.abs(data))
    maxInd = np.argmax(np.abs(data))
    for i, v in reversed(list(enumerate(data))):
        if v > percent*maxVal:
            minInd = i
            break

    return data[maxInd:minInd]

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

def get_period(data, sampleRate):
    '''Returns the period and uncertainty for data resembling a decaying
    oscillation.

    Parameters
    ----------
    data : ndarray, shape(n,)
        A time series that resembles a decaying oscillation.
    sampleRate : int
        The frequency that data was sampled at.

    Returns
    -------
    T : ufloat
        The period of oscillation and its uncertainty.

    '''

    y = data
    x = np.linspace(0., (len(y) - 1)/sampleRate, num=len(y))
    # decaying oscillating exponential function
    fitfunc = lambda p, t: p[0] + np.exp(-p[3]*p[4]*t)*(p[1]*np.sin(p[4]*np.sqrt(1-p[3]**2)*t) + p[2]*np.cos(p[4]*np.sqrt(1-p[3]**2)*t))
    #def fitfunc(p, t):
        #'''Decaying oscillation function.'''
        #a = p[0]
        #b = np.exp(-p[3] * p[4] * t)
        #c = p[1] * np.sin(p[4] * np.sqrt(1 - p[3]**2) * t)
        #d = p[2] * np.cos(p[4] * np.sqrt(1 - p[3]**2) * t)
        #return a + b * (c + d)
    # initial guesses
    #p0 = np.array([1.35, -.5, -.75, 0.01, 3.93])
    p0 = np.array([2.5, -.75, -.75, 0.001, 4.3])
    #p0 = make_guess(data, sampleRate)
    #print "guess:", p0
    # create the error function
    errfunc = lambda p, t, y: fitfunc(p, t) - y
    # minimize the error function
    p1, success = leastsq(errfunc, p0[:], args=(x, y))
    print p1, success
    lscurve = fitfunc(p1, x)
    #plt.plot(x, y, '.')
    #plt.plot(x, lscurve, '-')
    #plt.show()
    rsq, SSE, SST, SSR = fit_goodness(y, lscurve)
    #print rsq, SSE, SST, SSR
    sigma = np.sqrt(SSE / (len(y) - len(p0)))
    #print 'sigma', sigma
    # calculate the jacobian
    L = jac_fitfunc(p1, x)
    #print "L", L
    # the Hessian
    H = np.dot(L.T, L)
    #print "H", H
    #print "inv(H)", inv(H)
    # the covariance matrix
    U = sigma**2. * inv(H)
    #print "U", U
    # the standard deviations
    sigp = np.sqrt(U.diagonal())
    #print sigp
    # frequency and period
    wo = ufloat((p1[4], sigp[4]))
    zeta = ufloat((p1[3], sigp[3]))
    wd = (1. - zeta ** 2.) ** (1. / 2.) * wo
    f = wd / 2. / pi
    # return the period
    return 1. / f

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
    p[4] = (samplesPerPeriod/sampleRate/2./pi)**-1

    return p


def jac_fitfunc(p, t):
    '''
    Calculate the Jacobian of a decaying oscillation function.

    Uses the analytical formulations of the partial derivatives.

    Parameters:
    -----------
    p : the five parameters of the equation
    t : time vector

    Returns:
    --------
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
    jac[3] = -p[4] * t * e * (p[1] * s + p[2] * c) + e * (-p[1] * p[3] * p[4] * t / dampsq * c + p[2] * p[3] * p[4] * t / dampsq * s)
    jac[4] = -p[3] * t * e * (p[1] * s + p[2] * c) + e * dampsq * t * (p[1] * c - p[2] * s)
    return jac.T

def fit_goodness(ym, yp):
    '''
    Calculate the goodness of fit.

    Parameters:
    ----------
    ym : vector of measured values
    yp : vector of predicted values

    Returns:
    --------
    rsq: r squared value of the fit
    SSE: error sum of squares
    SST: total sum of squares
    SSR: regression sum of squares

    '''
    SSR = sum((yp - np.mean(ym))**2)
    SST = sum((ym - np.mean(ym))**2)
    SSE = SST - SSR
    rsq = SSR/SST
    return rsq, SSE, SST, SSR

def space_out_camel_case(s, output='string'):
        """Adds spaces to a camel case string.  Failure to space out string
        returns the original string.

        Examples
        --------
        >>> space_out_camel_case('DMLSServicesOtherBSTextLLC')
        'DMLS Services Other BS Text LLC'
        >>> space_out_camel_case('DMLSServicesOtherBSTextLLC', output='list')
        ['DMLS', 'Services', 'Other', 'BS', 'Text', 'LLC']

        """
        if output == 'string':
            return re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ',
                          s).strip()
        elif output == 'list':
            string = re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', ' ',
                            s).strip()
            return string.split(' ')
        else:
            raise ValueError

def filename_to_dict(filename):
    '''Returns a dictionay of values based on the pendulum data file name.

    '''
    o = space_out_camel_case(os.path.splitext(filename)[0], output='list')
    # this only accounts for single digit trial numbers
    trial = o[-1][-1]
    o[-1] = o[-1][:-1]
    o.append(trial)
    breakdown = ['bicycle', 'part', 'pendulum', 'angleOrder', 'trial']
    dat = {}
    for word, val  in zip(breakdown, o):
        dat[word] = val
    return dat

def load_parameter_text_file(pathToFile):
    '''Returns a dictionary of ufloat parameters from a parameter file.

    Parameters
    ----------
    pathToFile : string
        The path to the text file with the parameters listed in the specified
        format.

    Returns
    -------
    parameters : dictionary

    For example:

    'c = 0.08 +/- 0.01\nd=0.314+/-0.002\nt = 0.1+/-0.01, 0.12+/-0.02'

    The first item on the line must be the variable name and the second is an
    equals sign. The values to the right of the equal sign much contain an
    uncertainty and multiple comma seperated values will be averaged.

    '''

    f = open(pathToFile, 'r')
    parameters = {}
    # parse the text file
    for line in f:
        if line[0] != '#':
            # remove any whitespace characters and split into a list
            equality = line.strip().split('=')
            # ['a ', ' 0.1 +/- 0.05 , 0.09 +/- 0.05']
            vals = equality[1].strip().split(',')
            # ['0.1 +/- 0.05 ', ' 0.09 +/- 0.05']
            ufloats = [ufloat(x) for x in vals]
            parameters[equality[0].strip()] = np.mean(ufloats)

    return parameters

def load_pendulum_data_mat_file(pathToFile):
    '''Returns a dictionay containing the data from the pendulum data mat file.

    '''
    pendDat = {}
    loadmat(pathToFile, mdict=pendDat)
    #clean up the matlab imports
    del(pendDat['__globals__'], pendDat['__header__'], pendDat['__version__'])
    for k, v in pendDat.items():
        try:
            #change to an ascii string
            pendDat[k] = v[0].encode('ascii')
        except:
            #if an array of a single number
            if np.shape(v)[0] == 1:
                pendDat[k] = v[0][0]
            #else if the notes are empty
            elif np.shape(v)[0] == 0:
                pendDat[k] = ''
            #else it is the data which needs to be a one dimensional array
            else:
                pendDat[k] = v.reshape((len(v),))
    return pendDat

