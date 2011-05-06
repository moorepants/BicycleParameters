#!/usr/bin/env python
import os
import re
from math import pi

import numpy as np
from numpy import ma
from scipy.optimize import leastsq, newton
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from uncertainties import ufloat, unumpy, umath

class Bicycle(object):
    '''An object for a bicycle. A bicycle has parameters. That's about it for
    now.

    '''

    def __new__(cls, shortname, forceRawCalc=False, forcePeriodCalc=False):
        '''Returns a NoneType object if there is no directory for the bicycle.'''
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
            b = "bicycle in bicycles/%s. " % shortname
            c = "Then I can actually created a bicycle object."
            print a + b + c
            return None

    def __init__(self, shortname, forceRawCalc=False, forcePeriodCalc=False):
        '''Creates a bicycle object and sets the parameters based on the
        available data.

        Parameters
        ----------
        shortname : string
            The short name of your bicicleta. It should be one word with the
            first letter capitilized and all other letters lower case. You
            should have a matching directory under "bicycles/". For example:
            "bicycles/Shortname".

        forceRawCalc : boolean
            Forces a recalculation of the benchmark parameters from the measured
            parameter. Otherwise it will only run the calculation if there is
            no benchmark parameter file.

        '''

        self.shortname = shortname
        self.directory = os.path.join('bicycles', shortname)
        self.parameters = {}

        # if there are some parameter files, then load them
        if 'Parameters' in os.listdir(self.directory):
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

        rawDataDir = os.path.join(self.directory, 'RawData')

        # it would be more robust to see if there are enough files in the
        # RawData directory
        isRawDataDir = 'RawData' in os.listdir(self.directory)
        if isRawDataDir:
            isMeasuredFile = shortname + 'Measured.txt' in os.listdir(rawDataDir)
        else:
            isMeasuredFile = False
        isBenchmark = 'Benchmark' in self.parameters.keys()

        # the user wants to force a recalc and the data is there
        conOne = forceRawCalc and isRawDataDir and isMeasuredFile
        # the user doesn't want to force a recalc and there are no benchmark
        # parameters
        conTwo = not forceRawCalc and not isBenchmark

        if conOne or conTwo:
            calc = self.calculate_from_measured(forcePeriodCalc=forcePeriodCalc)
            par, slopes, intercepts, betas, pendulumInertias = calc
            self.parameters['Benchmark'] = par
            self.slopes = slopes
            self.intercepts = intercepts
            self.betas = betas
            self.pendulumInertias = pendulumInertias
            print("The glory of the %s parameters are upon you!"
                  % self.shortname)
        elif not forceRawCalc and isBenchmark:
            # we already have what we need
            stmt1 = "Looks like you've already got some parameters for %s, "
            stmt2 = "use forceRawCalc to recalculate."
            print (stmt1 + stmt2) % self.shortname
            pass
        else:
            print '''There is no data available. Create
            bicycles/{sn}/Parameters/{sn}Benchmark.txt and/or fill
            bicycle/{sn}/RawData/ with pendulum data mat files and the
            {sn}Measured.txt file'''.format(sn=shortname)

    def save_parameters(self, filetype='text', uncert=True):
        '''
        Saves all the parameters to file.

        filetype : string, optional
            'pickle' : python pickled dictionary
            'matlab' : matlab .mat file
            'text' : 

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

    def show_pendulum_photos(self):
        '''Opens up the pendulum photos in eye of gnome for inspection.

        This only works in Linux and if eog is installed. Maybe check pythons
        xdg-mime model for having this work cross platform.

        '''
        photoDir = os.path.join(self.directory, 'Photos', '*.*')
        os.system('eog ' + photoDir)

    def calculate_from_measured(self, forcePeriodCalc=False):
        '''Calculates the parameters from measured data.'''

        rawDataDir = os.path.join(self.directory, 'RawData')
        pathToRawFile = os.path.join(rawDataDir, self.shortname + 'Measured.txt')

        # load the measured parameters
        self.parameters['Measured'] = load_parameter_text_file(pathToRawFile)

        forkIsSplit = is_fork_split(self.parameters['Measured'])

        # if the the user doesn't specifiy to force period calculation, then
        # see if enough data is actually available in the *Measured.txt file to
        # do the calculations
        if not forcePeriodCalc:
            forcePeriodCalc = check_for_period(self.parameters['Measured'],
                                               forkIsSplit)

        if forcePeriodCalc == True:
            # get the list of mat files associated with this bike
            matFiles = [x for x in os.listdir(rawDataDir)
                        if x.endswith('.mat')]
            # calculate the period for each file for this bicycle
            periods = calc_periods_for_files(rawDataDir, matFiles, forkIsSplit)
            # add the periods to the measured parameters
            self.parameters['Measured'].update(periods)

            write_periods_to_file(pathToRawFile, periods)

        return calculate_benchmark_from_measured(self.parameters['Measured'])

    def plot_bicycle_geometry(self, show=True):
        '''Returns a figure showing the basic bicycle geometry, the centers of
        mass and the moments of inertia.

        '''
        par = self.parameters['Benchmark']
        slopes = self.slopes
        intercepts = self.intercepts

        fig = plt.figure()
        ax = plt.axes()
        # plot the rear wheel
        c = plt.Circle((0., par['rR'].nominal_value),
                       radius=par['rR'].nominal_value,
                       fill=False)
        ax.add_patch(c)
        # plot the front wheel
        c = plt.Circle((par['w'].nominal_value, par['rF'].nominal_value),
                       radius=par['rF'].nominal_value,
                       fill=False)
        ax.add_patch(c)
        # plot the pendulum axes for the measured parts
        numColors = len(slopes.keys())
        cmap = plt.get_cmap('gist_rainbow')
        comLineLength = par['w'].nominal_value / .4
        for j, pair in enumerate(slopes.items()):
            part, slopeSet = pair
            xcom = par['x' + part].nominal_value
            zcom = par['z' + part].nominal_value
            plt.plot(xcom, -zcom, 'k+', markersize=12)
            for i, m in enumerate(slopeSet):
                m = m.nominal_value
                #comLineLength = self.pendulumInertias[part][i].nominal_value
                xPlus = comLineLength / 2. * np.cos(np.arctan(m))
                x = np.array([xcom - xPlus,
                              xcom + xPlus])
                y = -m * x - intercepts[part][i].nominal_value
                plt.plot(x, y, color=cmap(1. * j / numColors))
                plt.text(x[0], y[0], str(i + 1))
        plt.plot(par['xH'].nominal_value,
                 -par['zH'].nominal_value, 'k+', markersize=12)
        # plot the ground line
        x = np.array([-par['rR'].nominal_value,
                      par['w'].nominal_value + par['rF'].nominal_value])
        plt.plot(x, np.zeros_like(x), 'k')
        # plot the fundamental bike
        deex, deez = fundamental_geometry_plot_data(par)
        plt.plot(deex, -deez, 'k')
        plt.axis('equal')
        plt.ylim((0, 1))
        plt.title(self.shortname)

        # plot the principal moments of inertia
        tensors = {}
        for part in slopes.keys():
            I = unumpy.nominal_values(inertia_tensor(par, part))
            tensors['I' + part] = principal_axes(I)

        if 'H' not in slopes.keys():
            I = unumpy.nominal_values(inertia_tensor(par, 'H'))
            tensors['IH'] = principal_axes(I)

        for tensor in tensors:
            print "Part", tensor
            Ip, C = tensors[tensor]
            part = tensor[1]
            center = unumpy.nominal_values(np.array([par['x' + part],
                                                     -par['z' + part]]))
            # which row in C is the y vector
            uy = np.array([0., 1., 0.])
            for i, row in enumerate(C):
                if np.abs(np.sum(row - uy)) < 1E-10:
                    yrow = i
            # the 3 is just random scaling factor
            Ip2D = np.delete(Ip, yrow, 0)
            # remove the column and row associated with the y
            C2D = np.delete(np.delete(C, yrow, 0), 1, 1)
            # make an ellipse
            height = Ip2D[0]
            width = Ip2D[1]
            angle = np.degrees(np.arccos(C2D[0, 0]))
            ellipse = Ellipse((center[0], center[1]), width, height,
                    angle=angle, fill=False)
            ax.add_patch(ellipse)

        if show:
            fig.show()

        return fig

def inertia_tensor(par, part):
    '''Returns an inertia tensor for a particular part for the benchmark
    parameter set.

    Parameters
    ----------
    par : dictionary
        Complete Benchmark parameter set.
    part : string
        Either 'B', 'H', 'F', 'R', 'G', 'S'

    Returns
    -------
    I : ndarray, shape(3,3)
        Inertia tensor for the part.

    '''
    I = np.zeros((3, 3), dtype=object)
    # front or rear wheel
    if part == 'F' or part == 'R':
        axes = np.array([['xx', None, None],
                         [None, 'yy', None],
                         [None, None, 'xx']])
    # all other parts
    else:
        axes = np.array([['xx', None, 'xz'],
                         [None, 'yy', None],
                         ['xz', None, 'zz']])
    for i, row in enumerate(axes):
        for j, col in enumerate(row):
            if col != None:
                I[i, j] = par['I' + part + col]
    return I

def fundamental_geometry_plot_data(par):
    '''Returns the coordinates for line end points of the bicycle fundamental
    geometry.

    Parameters
    ----------
    par : dictionary
        Benchmark bicycle parameters.

    Returns
    -------
    x : ndarray
    z : ndarray

    '''
    d1 = umath.cos(par['lam']) * (par['c'] + par['w'] -
                par['rR'] * umath.tan(par['lam']))
    d3 = -umath.cos(par['lam']) * (par['c'] - par['rF'] *
                umath.tan(par['lam']))
    x = np.zeros(4, dtype=object)
    z = np.zeros(4, dtype=object)
    x[0] = 0.
    x[1] = d1 * umath.cos(par['lam'])
    x[2] = par['w'] - d3 * umath.cos(par['lam'])
    x[3] = par['w']
    z[0] = -par['rR']
    z[1] = -par['rR'] - d1 * umath.sin(par['lam'])
    z[2] = -par['rF'] + d3 * umath.sin(par['lam'])
    z[3] = -par['rF']

    return unumpy.nominal_values(x), unumpy.nominal_values(z)

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
    pathToBicycleDir = os.path.join(pathToRawDataParts[0], pathToRawDataParts[1])
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

def write_periods_to_file(pathToRawFile, mp):
    '''Writes the provided periods to file.

    Parameters
    ----------
    pathToRawFile : string
        The path to the *Measured.txt file
    mp : dictionary
        The measured parameters dictionary. Should contain complete period
        data.

    '''

    # clear any period data from the file
    f = open(pathToRawFile, 'r')
    baseData = ''
    for line in f:
        if not line.startswith('T'):
            baseData += line
    f.close()
    # add the periods to the base data
    periodKeys = [x for x in mp.keys() if x.startswith('T')]
    periodKeys.sort()
    withPeriods = baseData
    for k in periodKeys:
        withPeriods += k + ' = ' + str(mp[k]) + '\n'

    # write it to the file
    f = open(pathToRawFile, 'w')
    f.write(withPeriods)
    f.close()

def calculate_benchmark_from_measured(mp):
    '''Returns the benchmark (Meijaard 2007) parameter set based on the
    measured data.

    Parameters
    ----------
    mp : dictionary
        Complete set of measured data.

    Returns
    -------
    par : dictionary
        Benchmark bicycle parameter set.

    '''

    forkIsSplit = is_fork_split(mp)

    par = {}

    # calculate the wheelbase, steer axis tilt and trail
    par = calculate_benchmark_geometry(mp, par)

    # masses
    par['mB'] = mp['mB']
    par['mF'] = mp['mF']
    par['mR'] = mp['mR']
    if forkIsSplit:
        par['mS'] = mp['mS']
        par['mG'] = mp['mG']
    else:
        par['mH'] = mp['mH']

    # get the slopes, intercepts and betas for each part
    slopes, intercepts, betas = part_com_lines(mp, par, forkIsSplit)

    # calculate the centers of mass
    for part in slopes.keys():
        par['x' + part], par['z' + part] = center_of_mass(slopes[part],
            intercepts[part])

    # find the center of mass of the handlebar/fork assembly if the fork was
    # split
    if forkIsSplit:
        coordinates = np.array([[par['xS'], par['xG']],
                                [0., 0.],
                                [par['zS'], par['zG']]])
        masses = np.array([par['mS'], par['mG']])
        mH, cH = total_com(coordinates, masses)
        par['mH'] = mH
        par['xH'] = cH[0]
        par['zH'] = cH[2]

    # calculate the stiffness of the torsional pendulum
    iRod = tube_inertia(mp['lP'], mp['mP'], mp['dP'] / 2., 0.)[1]
    torStiff = tor_stiffness(iRod, mp['TtP1'])
    print "Torsional stiffness of the rod:", torStiff

    # local accelation due to gravity
    par['g'] = mp['g']

    # calculate the wheel y inertias
    par['IFyy'] = compound_pendulum_inertia(mp['mF'], mp['g'],
                                            mp['lF'], mp['TcF1'])
    par['IRyy'] = compound_pendulum_inertia(mp['mR'], mp['g'],
                                            mp['lR'], mp['TcR1'])

    # calculate the wheel x/z inertias
    par['IFxx'] = tor_inertia(torStiff, mp['TtF1'])
    par['IRxx'] = tor_inertia(torStiff, mp['TtR1'])

    # calculate the y inertias for the frame and fork
    lB = (par['xB']**2 + (par['zB'] + par['rR'])**2)**(0.5)
    par['IByy'] = compound_pendulum_inertia(mp['mB'], mp['g'], lB, mp['TcB1'])

    if forkIsSplit:
        # fork
        lS = ((par['xS'] - par['w'])**2 +
              (par['zS'] + par['rF'])**2)**(0.5)
        par['ISyy'] = compound_pendulum_inertia(mp['mS'], mp['g'],
                                                lS, mp['TcS1'])
        # handlebar
        l1, l2 = calculate_l1_l2(mp['h6'], mp['h7'],
                                 mp['d5'], mp['d6'], mp['l'])
        u1, u2 = fwheel_to_handlebar_ref(par['lam'], l1, l2)
        lG = ((par['xG'] - par['w'] + u1)**2 +
              (par['zG'] + par['rF'] + u2)**2)**(.5)
        par['IGyy'] = compound_pendulum_inertia(mp['mG'], mp['g'],
                                                lG, mp['TcG1'])
    else:
        lH = ((par['xH'] - par['w'])**2 +
              (par['zH'] + par['rF'])**2)**(0.5)
        par['IHyy'] = compound_pendulum_inertia(mp['mH'], mp['g'],
                                                lH, mp['TcH1'])

    pendulumInertias = {}

    # calculate the in plane moments of inertia
    for part, slopeSet in slopes.items():
        print "The part is:", part
        # the number of orientations for this part
        numOrien = len(slopeSet)
        # intialize arrays to store the inertia values and orientation angles
        eye = np.zeros(numOrien, dtype=object)
        beta = np.array(betas[part])
        # fill arrays of the inertias and orientation angles
        for i in range(numOrien):
            eye[i] = tor_inertia(torStiff, mp['Tt' + part + str(i + 1)])
        pendulumInertias[part] = eye
        print "The pendulum inertias:\n", eye
        print "The orientations:\n", beta
        inertia = inertia_components(eye, beta)
        for i, axis in enumerate(['xx', 'xz', 'zz']):
            par['I' + part + axis] = inertia[i]

    if forkIsSplit:
        # combine the moments of inertia to find the total handlebar/fork MoI
        IG = inertia_tensor(par, 'G')
        IS = inertia_tensor(par, 'S')
        coordinates = np.array([[par['xG'], par['xS']],
                                [0., 0.],
                                [par['zG'], par['zS']]])
        masses = np.array([par['mG'], par['mS']])
        par['mH'], cH = total_com(coordinates, masses)
        par['xH'] = cH[0]
        par['zH'] = cH[2]
        dG = np.array([par['xG'] - par['xH'], 0., par['zG'] - par['zH']])
        dS = np.array([par['xS'] - par['xH'], 0., par['zS'] - par['zH']])
        IH = (parallel_axis(IG, par['mG'], dG) +
              parallel_axis(IS, par['mS'], dS))
        par['IHxx'] = IH[0, 0]
        par['IHxz'] = IH[0, 2]
        par['IHyy'] = IH[1, 1]
        par['IHzz'] = IH[2, 2]

    return par, slopes, intercepts, betas, pendulumInertias

def principal_axes(I):
    '''Returns the principal moments of inertia and the orientation.

    Parameters
    ----------
    I : ndarray, shape(3,3)
        An inertia tensor.

    Returns
    -------
    Ip : ndarray, shape(3,)
        The principal moments of inertia.
    C : ndarray, shape(3,3)
        The rotation matrix.

    '''
    Ip, C = np.linalg.eig(I)
    indices = np.argsort(Ip)
    Ip = Ip[indices]
    C = C.T[indices]
    return Ip, C

def parallel_axis(Ic, m, d):
    '''Returns the moment of inertia of a body about a different point.

    Parameters
    ----------
    Ic : ndarray, shape(3,3)
        The moment of inertia about the center of mass of the body with respect
        to an orthogonal coordinate system.
    m : float
        The mass of the body.
    d : ndarray, shape(3,)
        The distances along the three ordinates that located the new point
        relative to the center of mass of the body.

    Returns
    -------
    I : ndarray, shape(3,3)
        The moment of inertia about of the body about a point located by d.

    '''
    a = d[0]
    b = d[1]
    c = d[2]
    dMat = np.zeros((3, 3), dtype=object)
    dMat[0] = np.array([b**2 + c**2, -a * b, -a * c])
    dMat[1] = np.array([-a * b, c**2 + a**2, -b * c])
    dMat[2] = np.array([-a * c, -b * c, a**2 + b**2])
    return Ic + m * dMat

def inertia_components(jay, alpha):
    '''Returns the 2D orthogonal inertia tensor.

    When at least three moments of inertia and their axes orientations are
    known relative to a common inertial frame of a planar object, the orthoganl
    moments of inertia relative the frame are computed.

    Parameters
    ----------
    jay : ndarray, shape(n,)
        An array of at least three moments of inertia. (n >= 3)
    alpha : ndarray, shape(n,)
        An array of orientation angles corresponding to the moments of inertia
        in jay.

    Returns
    -------
    eye : ndarray, shape(3,)
        Ixx, Ixz, Izz

    '''
    sa = unumpy.sin(alpha)
    ca = unumpy.cos(alpha)
    a = unumpy.matrix(np.vstack((ca**2, -2 * sa * ca, sa**2)).T)
    eye = np.squeeze(np.asarray(np.dot(a.I, jay)))
    return eye

def tor_inertia(k, T):
    '''Calculate the moment of interia for an ideal torsional pendulm

    Parameters:
    -----------
    k: torsional stiffness
    T: period

    Returns:
    --------
    I: moment of inertia

    '''

    I = k * T**2 / 4. / pi**2

    return I

def compound_pendulum_inertia(m, g, l, T):
    '''Returns the moment of inertia for an object hung as a compound
    pendulum.

    Parameters
    ----------
    m : float
        Mass of the pendulum.
    g : float
        Acceration due to gravity.
    l : float
        Length of the pendulum.
    T : float
        The period of oscillation.

    Returns
    -------
    I : float
        Moment of interia of the pendulum.

    '''

    I = (T / 2. / pi)**2. * m * g * l - m * l**2.

    return I

def tor_stiffness(I, T):
    '''Calculate the stiffness of a torsional pendulum with a known moment of
    inertia.

    Parameters
    ----------
    I : moment of inertia
    T : period

    Returns
    -------
    k : stiffness

    '''
    k = 4. * I * pi**2 / T**2
    return k

def tube_inertia(l, m, ro, ri):
    '''Calculate the moment of inertia for a tube (or rod) where the x axis is
    aligned with the tube's axis

    Parameters
    ----------
    l: length
    m: mass
    ro: outer radius
    ri: inner radius

    Returns
    -------
    Ix: moment of inertia about tube axis
    Iy, Iz: moment of inertia about normal axis

    '''
    Ix = m / 2. * (ro**2 + ri**2)
    Iy = m / 12. * (3 * ro**2 + 3 * ri**2 + l**2)
    Iz = Iy
    return np.array([Ix, Iy, Iz])

def total_com(coordinates, masses):
    '''Returns the center of mass of a group of objects if the indivdual
    centers of mass and mass is provided.

    coordinates : ndarray, shape(3,n)
        The rows are the x, y and z coordinates, respectively and the columns
        are for each object.
    masses : ndarray, shape(3,)
        An array of the masses of multiple objects, the order should correspond
        to the columns of coordinates.

    Returns
    -------
    mT : float
        Total mass of the objects.
    cT : ndarray, shape(3,)
        The x, y, and z coordinates of the total center of mass.

    '''
    products = masses * coordinates
    mT = np.sum(masses)
    cT = np.sum(products, axis=1) / mT
    return mT, cT

def part_com_lines(mp, par, forkIsSplit):
    '''Returns the slopes and intercepts for all of the center of mass lines
    for each part.

    Parameters
    ----------
    mp : dictionary
        Dictionary with the measured parameters.

    Returns
    -------
    slopes : dictionary
        Contains a list of slopes for each part.
    intercepts : dictionary
        Contains a list of intercepts for each part.

    The slopes and intercepts lists are in order with respect to each other and
    the keyword is either 'B', 'H' or 'S'.

    '''
    # find the slope and intercept for pendulum axis
    if forkIsSplit:
        l1, l2 = calculate_l1_l2(mp['h6'], mp['h7'], mp['d5'], mp['d6'], mp['l'])
        slopes = {'B':[], 'G':[], 'S':[]}
        intercepts = {'B':[], 'G':[], 'S':[]}
        betas = {'B':[], 'G':[], 'S':[]}
    else:
        l1, l2 = 0., 0.
        slopes = {'B':[], 'H':[]}
        intercepts = {'B':[], 'H':[]}
        betas = {'B':[], 'H':[]}

    for key, val in mp.items():
        if key.startswith('alpha'):
            a = mp['a' + key[5:]]
            part = key[5]
            m, b, beta = com_line(val, a, par, part, l1, l2)
            slopes[key[5]].append(m)
            intercepts[key[5]].append(b)
            betas[key[5]].append(beta)

    return slopes, intercepts, betas

def center_of_mass(slopes, intercepts):
    '''Returns the center of mass relative to the slopes and intercepts
    coordinate system.

    Parameters
    ----------
    slopes : ndarray, shape(n,)
        The slope of every line used to calculate the center of mass.
    intercepts : ndarray, shape(n,)
        The intercept of every line used to calculate the center of mass.

    Returns
    -------
        x : float
            The abscissa of the center of mass.
        y : float
            The ordinate of the center of mass.

    '''
    num = range(len(slopes))
    allComb = cartesian((num, num))
    comb = []
    # remove doubles
    for row in allComb:
        if row[0] != row[1]:
            comb.append(row)
    comb = np.array(comb)

    # initialize the matrix to store the line intersections
    lineX = np.zeros((len(comb), 2), dtype='object')
    # for each line intersection...
    for j, row in enumerate(comb):
        sl = np.array([slopes[row[0]], slopes[row[1]]])
        a = unumpy.matrix(np.vstack((-sl, np.ones((2)))).T)
        b = np.array([intercepts[row[0]], intercepts[row[1]]])
        lineX[j] = np.dot(a.I, b)
    com = np.mean(lineX, axis=0)

    return com[0], com[1]

def calculate_l1_l2(h6, h7, d5, d6, l):
    '''Returns the distance along (l2) and perpendicular (l1) to the steer axis from the
    front wheel center to the handlebar reference point.

    Parameters
    ----------
    h6 : float
        Distance from the table to the top of the front axle.
    h7 : float
        Distance from the table to the top of the handlebar reference circle.
    d5 : float
        Diameter of the front axle.
    d6 : float
        Diameter of the handlebar reference circle.
    l : float
        Outer distance from the front axle to the handlebar reference circle.

    Returns
    -------
    l1 : float
        The distance from the front wheel center to the handlebar reference
        center perpendicular to the steer axis. The positive sense is if the
        handlebar reference point is more forward than the front wheel center
        relative to the steer axis normal.
    l2 : float
       The distance from the front wheel center to the handlebar reference
       center parallel to the steer axis. The positive sense is if the
       handlebar reference point is above the front wheel center with reference
       to the steer axis.

    '''
    r5 = d5 / 2.
    r6 = d6 / 2.
    l1 = h7 - h6 + r5 - r6
    l0 = l - r5 - r6
    gamma = umath.asin(l1 / l0)
    l2 = l0 * umath.cos(gamma)
    return l1, l2

def com_line(alpha, a, par, part, l1, l2):
    '''Returns the slope and intercept for the line that passes through the
    part's center of mass with reference to the benchmark bicycle coordinate
    system.

    Parameters
    ----------
    alpha : float
        The angle the head tube makes with the horizontal. When looking at the
        bicycle from the right side this is the angle between a vector point
        out upwards along the steer axis and the earth horizontal with the
        positve direction pointing from the left to the right. If the bike is
        in its normal configuration this would be 90 degrees plus the steer
        axis tilt (lambda).
    a : float
        The distance from the pendulum axis to a reference point on the part,
        typically the wheel centers. This is positive if the point falls to the
        left of the axis and negative otherwise.
    par : dictionary
        Benchmark parameters. Must include lam, rR, rF, w
    part : string
        The subscript denoting which part this refers to.
    l1, l2 : floats
        The location of the handlebar reference point relative to the front
        wheel center when the fork is split. This is measured perpendicular to
        and along the steer axis, respectively.

    Returns
    -------
    m : float
        The slope of the line in the benchmark coordinate system.
    b : float
        The z intercept in the benchmark coordinate system.

    '''

    # beta is the angle between the x bike frame and the x pendulum frame, rotation
    # about positive y
    beta = par['lam'] - alpha * pi / 180

    # calculate the slope of the center of mass line
    m = -umath.tan(beta)

    # calculate the z intercept
    # this the bicycle frame
    if part == 'B':
        b = -a / umath.cos(beta) - par['rR']
    # this is the fork (without handlebar) or the fork and handlebar combined
    elif part == 'S' or part == 'H':
        b = -a / umath.cos(beta) - par['rF'] + par['w'] * umath.tan(beta)
    # this is the handlebar (without fork)
    elif part == 'G':
        u1, u2 = fwheel_to_handlebar_ref(par['lam'], l1, l2)
        b = -a / umath.cos(beta) - (par['rF'] + u2) + (par['w'] - u1) * umath.tan(beta)
    else:
        raise

    return m, b, beta

def fwheel_to_handlebar_ref(lam, l1, l2):
    '''Returns the distance along the benchmark coordinates from the front
    wheel center to the handlebar reference center.

    Parameters
    ----------
    lam : float
        Steer axis tilt.
    l1, l2 : float
        The distance from the front wheel center to the handlebar refernce
        center perpendicular to and along the steer axis.

    Returns
    -------
    u1, u2 : float

    '''

    u1 = l2 * umath.sin(lam) - l1 * umath.cos(lam)
    u2 = u1 / umath.tan(lam) + l1 / umath.sin(lam)
    return u1, u2

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def calculate_benchmark_geometry(mp, par):
    '''Returns the wheelbase, steer axis tilt and the trail.

    Parameters
    ----------
    mp : dictionary
        Dictionary with the measured parameters.
    par : dictionary
        Dictionary with the benchmark parameters.

    Returns
    -------
    par : dictionary
        par with the benchmark geometry added.

    '''
    # calculate the wheel radii
    par['rF'] = mp['dF'] / 2./ pi / mp['nF']
    par['rR'] = mp['dR'] / 2./ pi / mp['nR']

    # calculate the frame/fork fundamental geometry
    if 'w' in mp.keys(): # if there is a wheelbase
        # steer axis tilt in radians
        par['lam'] = pi / 180. * (90. - mp['gamma'])
        # wheelbase
        par['w'] = mp['w']
        # fork offset
        forkOffset = mp['f']
    else:
        h = (mp['h1'], mp['h2'], mp['h3'], mp['h4'], mp['h5'])
        d = (mp['d1'], mp['d2'], mp['d3'], mp['d4'], mp['d'])
        a, b, c = calculate_abc_geometry(h, d)
        par['lam'] = lambda_from_abc(par['rF'], par['rR'], a, b, c)
        par['w'] = (a + b) * umath.cos(par['lam']) + c * umath.sin(par['lam'])
        forkOffset = b

    # trail
    par['c'] = trail(par['rF'], par['lam'], forkOffset)[0]

    return par

def calculate_abc_geometry(h, d):
    '''Returns the perpendicular distance geometry for the bicycle from the raw
    measurements.

    Parameters
    ----------
    h : tuple
        Tuple containing the measured parameters h1-h5.
        (h1, h2, h3, h4, h5)
    d : tuple
        Tuple containing the measured parameters d1-d4 and d.
        (d1, d2, d3, d4, d)

    Returns
    -------
    a : ufloat or float
        The rear frame offset.
    b : ufloat or float
        The fork offset.
    c : ufloat or float
        The steer axis distance.

    '''
    # extract the values
    h1, h2, h3, h4, h5 = h
    d1, d2, d3, d4, d = d
    # get the perpendicular distances
    a = h1 + h2 - h3 + .5 * d1 - .5 * d2
    b = h4 - .5 * d3 - h5 + .5 * d4
    c = umath.sqrt(-(a - b)**2 + (d + .5 * (d2 + d3)))
    return a, b, c

def get_sample_rate(matData):
    '''Returns the sample rate for the data.'''
    if 'ActualRate' in matData.keys():
        sampleRate = matData['ActualRate']
    else:
        sampleRate = matData['sampleRate']
    return sampleRate

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

def is_fork_split(mp):
    '''Returns true if the fork was split into two parts and false if not.

    Parameters
    ----------
    mp : dictionary
        The measured data.

    Returns
    -------
    forkIsSplit : boolean

    '''
    # this isn't that robust, for example if you had an S and no G then this
    # wouldn't catch it
    forkIsSplit = False
    for key in mp.keys():
        # if there is an 'S' then the fork is split in two parts
        if key[:1] == 'S' or key[1:2] == 'S':
            forkIsSplit = True

    return forkIsSplit

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
    c = (rF * umath.sin(lam) - fo) / umath.cos(lam)
    # mechanical trail
    cm = c * umath.cos(lam)
    return c, cm

def lambda_from_abc(rF, rR, a, b, c):
    '''Returns the steer axis tilt, lamba, for the parameter set based on the
    offsets from the steer axis.

    '''
    def lam_equality(lam, rF, rR, a, b, c):
        return umath.sin(lam) - (rF - rR + c * umath.cos(lam)) / (a + b)
    guess = umath.atan(c / (a + b)) # guess based on equal wheel radii

    # The following assumes that the uncertainty caluclated for the guess is
    # the same as the uncertainty for the true solution. This is not true! and
    # will surely breakdown the further the guess is away from the true
    # solution. There may be a way to calculate the correct uncertainity, but
    # that needs to be figured out. I guess I could use least squares and do it
    # the same way as get_period.

    args = (rF.nominal_value, rR.nominal_value, a.nominal_value,
            b.nominal_value, c.nominal_value)

    lam = newton(lam_equality, guess.nominal_value, args=args)
    return ufloat((lam, guess.std_dev()))

def get_period_from_truncated(data, sampleRate, pathToPlotFile):
    #dataRec = average_rectified_sections(data)
    dataRec = data
    dataGood = select_good_data(dataRec, 0.1)
    return get_period(dataGood, sampleRate, pathToPlotFile)

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
        A path to the directory for the plots.

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

def plot_osfit(t, ym, yf, p, rsq, T, m=None, fig=None):
    '''Plot fitted data over the measured

    Parameters:
    -----------
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

    Returns:
    --------
    fig : the figure

    '''
    # figure properties
    figwidth = 8. # in inches
    goldenMean = (np.sqrt(5)-1.0)/2.0
    figsize = [figwidth, figwidth*goldenMean]
    params = {#'backend': 'ps',
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'text.fontsize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        #'text.usetex': True,
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
    if m:
        plt.xlim((0, m))
    else:
        pass
    return fig

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

def load_pendulum_mat_file(pathToFile):
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
