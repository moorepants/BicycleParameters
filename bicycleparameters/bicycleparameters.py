#!/usr/bin/env python

# builtin modules
import os
import re
from math import pi

# dependencies
import numpy as np
from numpy import ma
from scipy.optimize import leastsq, newton, fmin
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge
from uncertainties import ufloat, unumpy, umath, UFloat
import yeadon

# local modules
from inertia import *

class Bicycle(object):
    """
    An object for a bicycle. A bicycle has parameters and can have a rider
    attached to it. That's about it for now.

    """

    def __new__(cls, bicycleName, pathToData=os.getcwd(), forceRawCalc=False,
            forcePeriodCalc=False):
        '''Returns a NoneType object if there is no directory for the bicycle.'''
        # is there a data directory for this bicycle? if not, tell the user to
        # put some data in the folder so we have something to work with!
        try:
            pathToBicycle = os.path.join(pathToData, 'bicycles', bicycleName)
            if os.path.isdir(pathToBicycle) == True:
                print("We have foundeth a directory named: " +
                      "{0}.".format(pathToBicycle))
                return super(Bicycle, cls).__new__(cls)
            else:
                raise ValueError
        except:
            a = "Are you nuts?! Make a directory called {0} ".format(bicycleName)
            b = "with basic data for your bicycle in {0}. ".format(pathToData)
            c = "Then I can actually created a bicycle object."
            print a + b + c
            return None

    def __init__(self, bicycleName, pathToData=os.getcwd(), forceRawCalc=False,
            forcePeriodCalc=False):
        """
        Creates a bicycle object and sets the parameters based on the available
        data.

        Parameters
        ----------
        bicycleName : string
            The short name of your bicicleta. It should be one word with the
            first letter capitalized and all other letters lower case. You
            should have a matching directory under `<pathToData>/bicycles/`.
            For example: `<pathToData>/bicycles/Shortname`.
        pathToData : string
            This is the path to the folder where the bicycle/rider parameters
            and raw data are stored. The default is the current working
            directory.
        forceRawCalc : boolean
            Forces a recalculation of the benchmark parameters from the measured
            parameters. Otherwise it will only run the calculation if there is
            no benchmark parameter file.
        forcePeriodCalc : boolean
            Forces a recalculation of the periods from the oscillation data.

        Notes
        -----
        Bicycles are assumed not to have a rider when initially loaded.

        """

        self.bicycleName = bicycleName
        pathToBicycles = os.path.join(pathToData, 'bicycles')
        # the directory where the files for this bicycle are stored
        self.directory = os.path.join(pathToBicycles, bicycleName)

        # bicycles are assumed not to have a rider when initially loaded
        self.hasRider = False
        self.riderPar = {}
        self.human = None

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

        # this is where the raw data files from the pendulum oscillations are
        # stored
        rawDataDir = os.path.join(self.directory, 'RawData')

        # it would be more robust to see if there are enough files in the
        # RawData directory, but that isn't implemented yet. For now you'll
        # just get and error sometime down the road when a period for the
        # missing files is needed.
        isRawDataDir = 'RawData' in os.listdir(self.directory)

        if isRawDataDir:
            print "Found the RawData directory:", rawDataDir
            isMeasuredFile = bicycleName + 'Measured.txt' in os.listdir(rawDataDir)
        else:
            isMeasuredFile = False

        isBenchmark = 'Benchmark' in self.parameters.keys()

        # the user wants to force a recalc and the data is there
        conOne = forceRawCalc and isRawDataDir and isMeasuredFile
        # the user doesn't want to force a recalc and there are no benchmark
        # parameters
        conTwo = not forceRawCalc and not isBenchmark

        if conOne or conTwo:
            print "Recalcuting the parameters."
            par, extras = self.calculate_from_measured(
                    forcePeriodCalc=forcePeriodCalc)
            self.parameters['Benchmark'] = par
            self.extras = extras
            print("The glory of the %s parameters are upon you!"
                  % self.bicycleName)
        elif not forceRawCalc and isBenchmark:
            # we already have what we need
            stmt1 = "Looks like you've already got some parameters for %s, "
            stmt2 = "use forceRawCalc to recalculate."
            print (stmt1 + stmt2) % self.bicycleName
            pass
        else:
            print '''There is no data available. Create
            bicycles/{sn}/Parameters/{sn}Benchmark.txt and/or fill
            bicycle/{sn}/RawData/ with pendulum data mat files and the
            {sn}Measured.txt file'''.format(sn=bicycleName)

    def __str__(self):
        if self.hasRider:
            desc = "{0} with {1} on board.".format(self.bicycleName,
                self.riderName)
        else:
            desc = "{0} with no one on board.".format(self.bicycleName)
        return desc

    def save_parameters(self, filetype='text'):
        """
        Saves all the parameter sets to file.

        Parameters
        ----------
        filetype : string, optional
            - 'text' : a text file with parameters as `c = 0.10+/-0.01\n`
            - 'matlab' : matlab .mat file
            - 'pickle' : python pickled dictionary

        """
        if self.hasRider:
            pathToData = os.path.split(os.path.split(self.directory)[0])[0]
            pathToParDir = os.path.join(pathToData, 'riders', self.riderName,
                                        'Parameters')
            pathToCombDir = os.path.join(pathToParDir, 'Combined')
            if not os.path.exists(pathToCombDir):
                os.makedirs(pathToCombDir)
            fileName = self.riderName + self.bicycleName
            # don't resave the measured parameters
            psets = [x for x in self.riderPar.keys() if x != 'Measured']
            parameters = self.riderPar
            print(('This bicycle has a rider, {0}, so the data will be ' +
                   'saved here: {1}').format(self.riderName, pathToParDir))
        else:
            pathToParDir = os.path.join(self.directory, 'Parameters')
            fileName = self.bicycleName
            # don't resave the measured parameters
            psets = [x for x in self.parameters.keys() if x != 'Measured']
            parameters = self.parameters
            print(('This bicycle has no rider so the data will be ' +
                   'saved here: {0}').format(pathToParDir))

        if filetype == 'text':
            for pset in psets:
                fileName = fileName + pset + '.txt'
                pathToTxtFile = os.path.join(pathToParDir, fileName)
                write_parameter_text_file(pathToTxtFile, parameters[pset])
                if self.hasRider:
                    pathToCombFile = os.path.join(pathToCombDir, fileName)
                    write_parameter_text_file(pathToCombFile,
                                              self.parameters[pset])

        elif filetype == 'matlab':
            # this should handle the uncertainties properly
            raise NotImplementedError("Doesn't work yet.")

        elif filetype == 'pickle':
            raise NotImplementedError("Doesn't work yet.")

    def show_pendulum_photos(self):
        """
        Opens up the pendulum photos in eye of gnome for inspection.

        This only works in Linux and if eog is installed. Maybe check pythons
        xdg-mime model for having this work cross platform.

        """
        photoDir = os.path.join(self.directory, 'Photos')
        if os.path.isdir(photoDir):
            os.system('eog ' + os.path.join(photoDir, '*.*'))
        else:
            print "There are no photos of your bicycle."

    def steer_assembly_moment_of_inertia(self, handlebar=True, fork=True,
            wheel=True, aboutSteerAxis=False, nominal=False):
        """
        Returns the inertia tensor of the steer assembly with respect to a
        reference frame aligned with the steer axis.

        Parameters
        ----------
        handlebar : boolean, optional
            If true the handlebar will be included in the calculation.
        fork : boolean, optional
            If true the fork will be included in the calculation.
        wheel : boolean, optional
            If true then the wheel will be included in the calculation.
        aboutSteerAxis : boolean, optional
            If true the inertia tensor will be with respect to a point made
            from the projection of the center of mass onto the steer axis.
        nominal : boolean, optional
            If true the nominal values will be returned instead of a uarray.

        Returns
        -------
        iAss : float
            Inertia tensor of the specified steer assembly parts with respect
            to a reference frame aligned with the steer axis.

        Notes
        -----
        The 3 component is aligned with the steer axis (pointing downward), the
        1 component is perpendicular to the steer axis (pointing forward) and
        the 2 component is perpendicular to the steer axis (pointing to the
        right).

        """
        # load in the Benchmark parameter set
        par = self.parameters['Benchmark']

        # there should always be either an H (handlebar/fork) and sometimes
        # there is a G (handlebar) and S (fork) if the fork and handlebar were
        # measured separately
        try:
            if fork and handlebar:
                # handlebar/fork
                I = part_inertia_tensor(par, 'H')
                m = par['mH']
                x = par['xH']
                z = par['zH']
            elif fork and not handlebar:
                # fork alone
                I = part_inertia_tensor(par, 'S')
                m = par['mS']
                x = par['xS']
                z = par['zS']
            elif handlebar and not fork:
                # handlebar alone
                I = part_inertia_tensor(par, 'G')
                m = par['mG']
                x = par['xG']
                z = par['zG']
            else:
                # if neither set to zero
                I = np.zeros((3, 3))
                m = 0.
                x = 0.
                z = 0.
        except KeyError:
            raise ValueError("The fork and handlebar were not measured " +
                             "separately for this bicycle." +
                             " Try making both the fork and handlebar either" +
                             " both True or both False.")

        if wheel:
            # list the mass and com of the handlebar/assembly and the front
            # wheel
            masses = np.array([m, par['mF']])

            coords = np.array([[x, par['w']],
                               [0., 0.],
                               [z, -par['rF']]])

            # mass and com of the entire assembly
            mAss, cAss = total_com(coords, masses)

            # front wheel inertia in the benchmark reference frame about the
            # com
            IF = part_inertia_tensor(par, 'F')

            # distance from the fork/handlebar assembly (without wheel) to the
            # new center of mass for the assembly with the wheel
            d = np.array([x - cAss[0], 0., z - cAss[2]])

            # distance from the front wheel center to the new center of mass
            # for the assembly with the wheel
            dF = np.array([par['w'] - cAss[0],
                           0.,
                           -par['rF'] - cAss[2]])

            # this is the inertia of the assembly about the com with reference
            # to the benchmark bicycle reference frame
            iAss = (parallel_axis(I, m, d) +
                    parallel_axis(IF, par['mF'], dF))

            # this is the inertia of the assembly about a reference frame aligned with
            # the steer axis and through the center of mass
            iAssRot = rotate_inertia_tensor(iAss, par['lam'])

        else: # don't add the wheel
            mAss = m
            cAss = np.array([x, 0., z])
            iAssRot = rotate_inertia_tensor(I, par['lam'])

        if aboutSteerAxis:
            # now find the inertia about the steer axis
            pointOnAxis1 = np.array([par['w'] + par['c'],
                                     0.,
                                     0.])
            pointOnAxis2 = pointOnAxis1 +\
                           np.array([-umath.sin(par['lam']),
                                     0.,
                                     -umath.cos(par['lam'])])
            pointsOnLine = np.array([pointOnAxis1, pointOnAxis2]).T

            # this is the distance from the assembly com to the steer axis
            distance = point_to_line_distance(cAss, pointsOnLine)
            print "handlebar cg distance", distance

            # now calculate the inertia about the steer axis of the rotated frame
            iAss = parallel_axis(iAssRot, mAss, np.array([distance, 0., 0.]))
        else:
            iAss = iAssRot

        if nominal:
            return unumpy.nominal_values(iAss)
        else:
            return iAss

    def calculate_from_measured(self, forcePeriodCalc=False):
        '''Calculates the parameters from measured data.'''

        rawDataDir = os.path.join(self.directory, 'RawData')
        pathToRawFile = os.path.join(rawDataDir, self.bicycleName + 'Measured.txt')

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
            matFiles.sort()
            # calculate the period for each file for this bicycle
            periods = calc_periods_for_files(rawDataDir, matFiles, forkIsSplit)
            # add the periods to the measured parameters
            self.parameters['Measured'].update(periods)

            write_periods_to_file(pathToRawFile, periods)

        return calculate_benchmark_from_measured(self.parameters['Measured'])

    def add_rider(self, rider, reCalc=False, draw=False):
        """
        Adds the inertial effects of a rigid rider to the bicycle.

        Parameters
        ----------
        rider : string
            A rider name that corresponds to a folder in
            `<pathToData>/riders/`.
        reCalc : boolean, optional
            If true, the rider parameters will be recalculated.
        draw : boolean, optional
            If true, visual python will be used to draw a three dimensional
            image of the rider.

        """

        # can't draw the rider model without the human object
        if draw:
            reCalc=True

        # first check to see if a rider has already been added
        if self.hasRider == True:
            print(("D'oh! This bicycle already has {0} as a " +
                  "rider!").format(self.riderName))
        else:
            print("There is no rider on the bicycle, now adding " +
                  "{0}.".format(rider))
            pathToData = os.path.split(os.path.split(self.directory)[0])[0]
            # get the path to the rider's folder
            pathToRider = os.path.join(pathToData, 'riders', rider)
            # load in the parameters
            bicyclePar = self.parameters['Benchmark']
            bicycle = self.bicycleName

            if reCalc == True:
                print("Calculating the human configuration.")
                # run the calculations
                try:
                    measuredPar = self.parameters['Measured']
                except KeyError:
                    print('The measured bicycle parameters need to be ' +
                          'available, create your bicycle such that they ' +
                          'are available.')
                    raise
                riderPar, human, bicycleRiderPar =\
                    configure_rider(pathToRider, bicycle, bicyclePar,
                        measuredPar, draw)
            else:
                pathToParFile = os.path.join(pathToRider, 'Parameters',
                    rider + self.bicycleName + 'Benchmark.txt')
                try:
                    # load the parameter file
                    riderPar = load_parameter_text_file(pathToParFile)
                except IOError:
                    # file doesn't exist so run the calculations
                    print("No parameter files found, calculating the human " +
                          "configuration.")
                    try:
                        measuredPar = self.parameters['Measured']
                    except KeyError:
                        print('The measured bicycle parameters need to be ' +
                              'available, create your bicycle such that they ' +
                              'are available.')
                        raise
                    riderPar, human, bicycleRiderPar =\
                        configure_rider(pathToRider, bicycle, bicyclePar,
                            measuredPar, draw)
                else:
                    print("Loaded the precalculated parameters from " +
                          "{0}".format(pathToParFile))
                    bicycleRiderPar = combine_bike_rider(bicyclePar, riderPar)
            # set the attributes
            self.riderPar['Benchmark'] = riderPar
            try:
                self.human = human
            except NameError:
                self.human = None
            self.parameters['Benchmark'] = bicycleRiderPar
            self.riderName = rider
            self.hasRider = True

    def plot_bicycle_geometry(self, show=True, pendulum=True,
                              centerOfMass=True, inertiaEllipse=True):
        '''Returns a figure showing the basic bicycle geometry, the centers of
        mass and the moments of inertia.

        '''
        par = remove_uncertainties(self.parameters['Benchmark'])
        parts = get_parts_in_parameters(par)

        try:
            slopes = remove_uncertainties(self.extras['slopes'])
            intercepts = remove_uncertainties(self.extras['intercepts'])
            penInertias = remove_uncertainties(self.extras['pendulumInertias'])
        except AttributeError:
            pendulum = False

        fig = plt.figure()
        ax = plt.axes()

        # define some colors for the parts
        numColors = len(parts)
        cmap = plt.get_cmap('gist_rainbow')
        partColors = {}
        for i, part in enumerate(parts):
            partColors[part] = cmap(1. * i / numColors)

        if inertiaEllipse:
            # plot the principal moments of inertia
            for j, part in enumerate(parts):
                I = part_inertia_tensor(par, part)
                Ip, C = principal_axes(I)
                if part == 'R':
                    center = np.array([0., par['rR']])
                elif part == 'F':
                    center = np.array([par['w'], par['rF']])
                else:
                    center = np.array([par['x' + part], -par['z' + part]])
                # which row in C is the y vector
                uy = np.array([0., 1., 0.])
                for i, row in enumerate(C):
                    if np.abs(np.sum(row - uy)) < 1E-10:
                        yrow = i
                # remove the row for the y vector
                Ip2D = np.delete(Ip, yrow, 0)
                # remove the column and row associated with the y
                C2D = np.delete(np.delete(C, yrow, 0), 1, 1)
                # make an ellipse
                Imin =  Ip2D[0]
                Imax = Ip2D[1]
                # get width and height of a ellipse with the major axis equal
                # to one
                unitWidth = 1. / 2. / np.sqrt(Imin) * np.sqrt(Imin)
                unitHeight = 1. / 2. / np.sqrt(Imax) * np.sqrt(Imin)
                # now scaled the width and height relative to the maximum
                # principal moment of inertia
                width = Imax * unitWidth
                height = Imax * unitHeight
                angle = -np.degrees(np.arccos(C2D[0, 0]))
                ellipse = Ellipse((center[0], center[1]), width, height,
                                  angle=angle, fill=False,
                                  color=partColors[part], alpha=0.25)
                ax.add_patch(ellipse)

        # plot the ground line
        x = np.array([-par['rR'],
                      par['w'] + par['rF']])
        plt.plot(x, np.zeros_like(x), 'k')

        # plot the rear wheel
        c = plt.Circle((0., par['rR']), radius=par['rR'], fill=False)
        ax.add_patch(c)

        # plot the front wheel
        c = plt.Circle((par['w'], par['rF']), radius=par['rF'], fill=False)
        ax.add_patch(c)

        # plot the fundamental bike
        deex, deez = fundamental_geometry_plot_data(par)
        plt.plot(deex, -deez, 'k')

        # plot the steer axis
        dx3 = deex[2] + deez[2] * (deex[2] - deex[1]) / (deez[1] - deez[2])
        plt.plot([deex[2], dx3],  [-deez[2], 0.], 'k--')

        # don't plot the pendulum lines if a rider has been added because the
        # inertia has changed
        if self.hasRider:
            pendulum = False

        if pendulum:
            # plot the pendulum axes for the measured parts
            for j, pair in enumerate(slopes.items()):
                part, slopeSet = pair
                xcom, zcom = par['x' + part], par['z' + part]
                for i, m in enumerate(slopeSet):
                    b = intercepts[part][i]
                    xPoint, zPoint = project_point_on_line((m, b), (xcom, zcom))
                    comLineLength = penInertias[part][i]
                    xPlus = comLineLength / 2. * np.cos(np.arctan(m))
                    x = np.array([xPoint - xPlus,
                                  xPoint + xPlus])
                    z = -m * x - b
                    plt.plot(x, z, color=partColors[part])
                    # label the pendulum lines with a number
                    plt.text(x[0], z[0], str(i + 1))

        if centerOfMass:
            # plot the center of mass location
            def com_symbol(ax, center, radius, color='b'):
                '''Returns axis with center of mass symbol.'''
                c = plt.Circle(center, radius=radius, fill=False)
                w1 = Wedge(center, radius, 0., 90.,
                           color=color, ec=None, alpha=0.5)
                w2 = Wedge(center, radius, 180., 270.,
                           color=color, ec=None, alpha=0.5)
                ax.add_patch(w1)
                ax.add_patch(w2)
                ax.add_patch(c)
                return ax

            # radius of the CoM symbol
            sRad = 0.03
            # front wheel CoM
            ax = com_symbol(ax, (par['w'], par['rF']), sRad,
                            color=partColors['F'])
            plt.text(par['w'] + sRad, par['rF'] + sRad, 'F')
            # rear wheel CoM
            ax = com_symbol(ax, (0., par['rR']), sRad,
                            color=partColors['R'])
            plt.text(0. + sRad, par['rR'] + sRad, 'R')
            for j, part in enumerate([x for x in parts
                                      if x != 'R' and x != 'F']):
                xcom = par['x' + part]
                zcom = par['z' + part]
                ax = com_symbol(ax, (xcom, -zcom), sRad,
                                color=partColors[part])
                plt.text(xcom + sRad, -zcom + sRad, part)
            if 'H' not in parts:
                ax = com_symbol(ax, (par['xH'], -par['zH']), sRad)
                plt.text(par['xH'] + sRad, -par['zH'] + sRad, 'H')


        plt.axis('equal')
        plt.ylim((0., 1.))
        plt.title(self.bicycleName)

        # if there is a rider on the bike, make a simple stick figure
        if self.human:
            human = self.human
            # K2: lower leg
            plt.plot([human.k[7].pos[0, 0], human.K2.pos[0, 0]],
                     [-human.k[7].endpos[2, 0], -human.K2.pos[2, 0]], 'k')
            # K1: upper leg
            plt.plot([human.K2.pos[0, 0], human.K1.pos[0, 0]],
                     [-human.K2.pos[2, 0], -human.K1.pos[2, 0]], 'k')
            # torso
            plt.plot([human.K1.pos[0, 0], human.B1.pos[0, 0]],
                     [-human.K1.pos[2, 0], -human.B1.pos[2, 0]], 'k')
            # B1: upper arm
            plt.plot([human.B1.pos[0, 0], human.B2.pos[0, 0]],
                     [-human.B1.pos[2, 0], -human.B2.pos[2, 0]], 'k')
            # B2: lower arm
            plt.plot([human.B2.pos[0, 0], human.b[6].pos[0, 0]],
                     [-human.B2.pos[2, 0], -human.b[6].endpos[2, 0]], 'k')
            # C: chest/head
            plt.plot([human.B1.pos[0, 0], human.C.endpos[0, 0]],
                     [-human.B1.pos[2, 0], -human.C.endpos[2, 0]], 'k')

        if show:
            fig.show()

        return fig

    def canonical(self):
        """
        Returns the canonical velocity and gravity independent matrices for the
        Whipple bicycle model.

        Returns
        -------
        M : ndarray, shape(2,2)
            Mass matrix.
        C1 : ndarray, shape(2,2)
            Velocity independent damping matrix.
        K0 : ndarray, shape(2,2)
            Gravity independent part of the stiffness matrix.
        K2 : ndarray, shape(2,2)
            Velocity squared independent part of the stiffness matrix.

        Notes
        -----

        The canonical matrices complete the following equation:

            M * q'' + v * C1 * q' + [g * K0 + v**2 * K2] * q = f

        where:

            q = [phi, delta]
            f = [Tphi, Tdelta]

        phi
            Bicycle roll angle.
        delta
            Steer angle.
        Tphi
            Roll torque.
        Tdelta
            Steer torque.
        v
            Bicylce speed.

        """

        par = self.parameters['Benchmark']

        M, C1, K0, K2 = benchmark_par_to_canonical(par)

        return M, C1, K0, K2

    def state_space(self, speed):
        """
        Returns the A and B matrices for the Whipple model linearized about
        the upright constant velocity configuration.


        Parameters
        ----------
        speed : float
            The speed of the bicycle.

        Returns
        -------

        A : ndarray, shape(4,4)
            The state matrix.
        B : ndarray, shape(4,2)
            The input matrix.

        Notes
        -----
        ``A`` and ``B`` describe the Whipple model in state space form:

            x' = A * x + B * u

        where

        The states are [roll rate,
                        steer rate,
                        roll angle,
                        steer angle]

        The inputs are [roll torque,
                        steer torque]

        """

        M, C1, K0, K2 = self.canonical()

        g = self.parameters['Benchmark']['g']

        A, B = abMatrix(M, C1, K0, K2, speed, g)

        return A, B

    def eig(self, speeds):
        '''Returns eigenvalues and eigenvectors of the benchmark bicycle.

        Parameters
        ----------
        speeds : ndarray, shape (n,) or float
            The speed at which to calculate the eigenvalues.

        Returns
        -------
        evals : ndarray, shape (n, 4)
            eigenvalues
        evecs : ndarray, shape (n, 4, 4)
            eigenvectors

        '''
        # this allows you to enter a float
        try:
            speeds.shape
        except AttributeError:
            speeds = np.array([speeds])

        m, n = 4, speeds.shape[0]
        evals = np.zeros((n, m), dtype='complex128')
        evecs = np.zeros((n, m, m), dtype='complex128')
        for i, speed in enumerate(speeds):
            A, B = self.state_space(speed)
            w, v = np.linalg.eig(unumpy.nominal_values(A))
            evals[i] = w
            evecs[i] = v

        return evals, evecs

    def plot_eigenvalues_vs_speed(self, speeds, fig=None, generic=False,
                                  color='black', show=True, largest=False,
                                  linestyle='-'):
        '''Returns a plot of the eigenvalues versus speed for the current
        benchmark parameters.

        Parameters
        ----------
        speeds : ndarray, shape(n,)
            An array of speeds to calculate the eigenvalues at.
        fig : matplotlib figure, optional
            A figure to plot to.
        generic : boolean
            If true the lines will all be the same color and the modes will not
            be labeled.
        color : matplotlib color
            If generic is true this will be the color of the plot lines.
        largest : boolean
            If true, only the largest eigenvalue is plotted.

        '''

        # sort the speeds in case they aren't
        speeds = np.sort(speeds)

        # figure properties
        figwidth = 6. # in inches
        goldenMean = (np.sqrt(5.)-1.) / 2.
        figsize = [figwidth, figwidth * goldenMean]
        params = {#'backend': 'ps',
            'axes.labelsize': 8,
            'text.fontsize': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'figure.figsize': figsize
            }
        plt.rcParams.update(params)

        if not fig:
            fig = plt.figure(figsize=figsize)

        plt.axes([0.125, 0.2, 0.95-0.125, 0.85-0.2])

        evals, evecs = self.eig(speeds)

        if largest:
            generic = True

        if generic:
            weaveColor = color
            capsizeColor = color
            casterColor = color
            legend = ['_nolegend_'] * 6
            legend[5] = self.bicycleName
            maxLabel = self.bicycleName
        else:
            weaveColor = 'blue'
            capsizeColor = 'red'
            casterColor = 'green'
            legend = ['Imaginary Weave', 'Imaginary Capsize',
                      'Imaginary Caster', 'Real Weave', 'Real Capsize',
                      'Real Caster']
            maxLabel = 'Max Eigenvalue'

        if largest:
            maxEval = np.max(np.real(evals), axis=1)
            plt.plot(speeds, maxEval, color=color, label=maxLabel,
                     linestyle=linestyle, linewidth=1.5)
            # x axis line
            plt.plot(speeds, np.zeros_like(speeds), 'k-',
                     label='_nolegend_', linewidth=1.5)
            plt.ylim((np.min(maxEval), np.max(maxEval)))
            plt.ylabel('Real Part of the Largest Eigenvalue [1/s]')
        else:
            wea, cap, cas = sort_modes(evals, evecs)

            # imaginary components
            plt.plot(speeds, np.abs(np.imag(wea['evals'])), color=weaveColor,
                     label=legend[0], linestyle='--')
            plt.plot(speeds, np.abs(np.imag(cap['evals'])), color=capsizeColor,
                     label=legend[1], linestyle='--')
            plt.plot(speeds, np.abs(np.imag(cas['evals'])), color=casterColor,
                     label=legend[2], linestyle='--')

            # x axis line
            plt.plot(speeds, np.zeros_like(speeds), 'k-',
                     label='_nolegend_', linewidth=1.5)

            # plot the real parts of the eigenvalues
            plt.plot(speeds, np.real(wea['evals']),
                     color=weaveColor, label=legend[3])
            plt.plot(speeds, np.real(cap['evals']),
                     color=capsizeColor, label=legend[4])
            plt.plot(speeds, np.real(cas['evals']),
                     color=casterColor, label=legend[5])

            # set labels and limits
            plt.ylim((np.min(np.real(evals)),
                      np.max(np.imag(evals))))
            plt.ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')

        plt.xlim((speeds[0], speeds[-1]))
        plt.xlabel('Speed [m/s]')

        if generic:
            plt.title('Eigenvalues vs Speed')
        else:
            plt.title('%s\nEigenvalues vs Speed' % self.bicycleName)
            plt.legend()

        if show:
            plt.show()

        return fig

def configure_rider(pathToRider, bicycle, bicyclePar, measuredPar, draw):
    """
    Returns the rider parameters, bicycle paramaters with a rider and a
    human object that is conifigured to sit on the bicycle.

    Parameters
    ----------
    pathToRider : string
        Path to the rider's data folder.
    bicycle : string
        The short name of the bicycle.
    bicyclePar : dictionary
        Contains the benchmark bicycle parameters for a bicycle.
    measuredPar : dictionary
        Contains the measured values of the bicycle.
    draw : boolean, optional
        If true, visual python will be used to draw a three dimensional
        image of the rider.

    Returns
    -------
    riderpar : dictionary
        The inertial parameters of the rider with reference to the
        benchmark coordinate system.
    human : yeadon.human
        The human object that represents the rider seated on the
        bicycle.
    bicycleRiderPar : dictionary
        The benchmark parameters of the bicycle with the rider added to
        the rear frame.

    """
    try:
        # get the rider name
        rider = os.path.split(pathToRider)[1]
        # get the paths to the yeadon data files
        pathToYeadon = os.path.join(pathToRider, 'RawData',
                                    rider + 'YeadonMeas.txt')
        pathToCFG = os.path.join(pathToRider, 'RawData',
                                 rider + bicycle + 'YeadonCFG.txt')
        # generate the human that has been configured to sit on the bicycle
        human = rider_on_bike(bicyclePar, measuredPar,
                              pathToYeadon, pathToCFG, draw)
        # build a dictionary to store the inertial data
        riderPar = {'IBxx': human.Inertia[0, 0],
                    'IByy': human.Inertia[1, 1],
                    'IBzz': human.Inertia[2, 2],
                    'IBxz': human.Inertia[2, 0],
                    'mB': human.Mass,
                    'xB': human.COM[0][0],
                    'yB': human.COM[1][0],
                    'zB': human.COM[2][0]}
    except: #except if this fails
        # no rider was added
        print('Calculations in yeadon failed. No rider added.')
        # raise the error that caused things to fail
        raise
    else:
        bicycleRiderPar = combine_bike_rider(bicyclePar, riderPar)
        return riderPar, human, bicycleRiderPar

def rider_on_bike(benchmarkPar, measuredPar, yeadonMeas, yeadonCFG,
                  drawrider):
    """
    Returns a yeadon human configured to sit on a bicycle.

    Parameters
    ----------
    benchmarkPar : dictionary
        A dictionary containing the benchmark bicycle parameters.
    measuredPar : dictionary
        A dictionary containing the raw geometric measurements of the bicycle.
    yeadonMeas : str
        Path to a text file that holds the 95 yeadon measurements. See
        `yeadon documentation`_.
    yeadonCFG : str
        Path to a text file that holds configuration variables. See `yeadon
        documentation`_. As of now, only 'somersalt' angle can be set as an
        input. The remaining variables are either zero or calculated in this
        method.
    drawrider : bool
        Switch to draw the rider, with vectors pointing to the desired
        position of the hands and feet of the rider (at the handles and
        bottom bracket). Requires python-visual.

    Returns
    -------
    H : yeadon.human
        Human object is returned with an updated configuration.
        The dictionary, taken from H.CFG, has the following key's values
        updated: ``CA1elevation``, ``CA1abduction``, ``A1A2flexion``,
        ``CB1elevation``, ``CB1abduction``, ``B1B2flexion``, ``PJ1elevation``,
        ``PJ1abduction``, ``J1J2flexion``, ``PK1elevation``, ``PK1abduction``,
        ``K1K2flexion``.

    Notes
    -----
    Requires that the bike object has a raw data text input file that contains
    the measurements necessary to situate a rider on the bike (i.e.
    ``<pathToData>/bicycles/<short name>/RawData/<short name>Measurements.txt``).

    .. _yeadon documentation : http://packages.python.org/yeadon


    """

    # create human using input measurements and configuration files
    H = yeadon.human(yeadonMeas, yeadonCFG)

    measuredPar = remove_uncertainties(measuredPar)
    benchmarkPar = remove_uncertainties(benchmarkPar)

    # for simplicity of code
    CFG = H.CFG
    # bottom bracket height
    hbb = measuredPar['hbb'] #.295
    # chain stay length
    Lcs = measuredPar['lcs'] #.46
    # rear wheel radius
    rR = benchmarkPar['rR'] #.342
    # front wheel radius
    rF = benchmarkPar['rF'] #.342
    # seat post length
    Lsp = measuredPar['lsp'] #.24
    # seat tube length
    Lst = measuredPar['lst'] #.53
    # seat tube angle
    lamst = measuredPar['lamst'] #68.5*np.pi/180
    # handlebar width
    whb = measuredPar['whb'] #43
    # distance from rear wheel hub to hand
    LhbR = measuredPar['LhbR'] #106
    # distance from front wheel hub to hand
    LhbF = measuredPar['LhbF'] #49
    # wheelbase
    w = benchmarkPar['w']

    # intermediate quantities
    # distance between wheel centers
    D = np.sqrt(w**2 + (rR - rF)**2)
    # projection into the plane of the bike
    dhbR = np.sqrt(LhbR**2 - (whb / 2)**2)
    dhbF = np.sqrt(LhbF**2 - (whb / 2)**2)
    # angle with vertex at rear hub, from horizontal "down" to front hub
    alpha = np.arcsin( (rR - rF) / D )
    # angle at rear hub of the dhbR-dhbF-D triangle (side-side-side)
    gamma = np.arccos( (dhbR**2 + D**2 - dhbF**2) / (2 * dhbR * D) )
    # position of bottom bracket center with respect to rear wheel contact
    # point
    pos_bb = np.array([[np.sqrt(Lcs**2 - (rR - hbb)**2)],
                       [0.],
                       [-hbb]])
    # vector from bottom bracket to seat
    vec_seat = -(Lst + Lsp) * np.array([[np.cos(lamst)],
                                        [0.],
                                        [np.sin(lamst)]])
    # position of seat with respect to rear wheel contact point
    pos_seat = pos_bb + vec_seat
    # vector (out of plane) from plane to right hand on the handlebars
    vec_hb_out  = np.array([[0.],
                            [whb / 2.],
                            [0.]])
    # vector (in plane) from rear wheel contact point to in-plane
    # location of hands
    vec_hb_in = np.array([[dhbR * np.cos(gamma - alpha)],
                          [0.],
                          [-rR - dhbR * np.sin(gamma - alpha)]])
    # position of right hand with respect to rear wheel contact point
    pos_handr = vec_hb_out + vec_hb_in
    # position of left hand with respect to rear wheel contact point
    pos_handl = -vec_hb_out + vec_hb_in

    # time to calculate the relevant quantities!
    # vector from seat to feet, ignoring out-of-plane distance
    vec_legs = -vec_seat
    # translation is done in bike's coordinate system
    H.translate_coord_sys(pos_seat)
    H.rotate_coord_sys((np.pi, 0., -np.pi /2.))
    # left foot
    pos_footl = pos_bb.copy()
    # set the y value at the same width as the hip
    pos_footl[1, 0] = H.J1.pos[1, 0]
    # right foot
    pos_footr = pos_bb.copy()
    # set the y value at the same width as the hip
    pos_footr[1, 0] = H.K1.pos[1, 0]

    # find the distance from the hip joint to the desired position of the foot
    DJ = np.linalg.norm( pos_footl - H.J1.pos)
    DK = np.linalg.norm( pos_footr - H.K1.pos)
    # find the distance from the should joint to the desired position of the
    # hands
    DA = np.linalg.norm( pos_handl - H.A1.pos)
    DB = np.linalg.norm( pos_handr - H.B1.pos)

    # distance from knees to heel level
    dj2 = np.linalg.norm( H.j[7].pos - H.J2.pos)
    dk2 = np.linalg.norm( H.k[7].pos - H.K2.pos)

    # distance from elbow to knuckle level
    da2 = np.linalg.norm( H.a[6].pos - H.A2.pos)
    db2 = np.linalg.norm( H.b[6].pos - H.B2.pos)

    # error-checking to make sure limbs are long enough for rider to sit
    # on the bike
    if (H.J1.length + dj2 < DJ):
        print "For the given measurements, the left leg is not " \
              "long enough. Left leg length is",H.J1.length+dj2, \
              "m, but distance from left hip joint to bottom bracket is", \
              DJ,"m."
        raise Exception()
    if (H.K1.length + dk2 < DK):
        print "For the given measurements, the right leg is not " \
              "long enough. Right leg length is",H.K1.length+dk2, \
              "m, but distance from right hip joint to bottom bracket is", \
              DK,"m."
        raise Exception()
    if (H.A1.length + da2 < DA):
        print "For the given configuration, the left arm is not " \
              "long enough. Left arm length is", H.A1.length + da2, \
              "m, but distance from shoulder to left hand is",DA,"m."
        raise Exception()
    if (H.B1.length + db2 < DB):
        print "For the given configuration, the right arm is not " \
              "long enough. Right arm length is",H.B1.length+db2, \
              "m, but distance from shoulder to right hand is",DB,"m."
        raise Exception()

    # joint angle time
    # legs first. torso cannot have twist
    # left leg
    tempangle, CFG['J1J2flexion'] =\
        calc_two_link_angles(H.J1.length, dj2, DJ)
    tempangle2 = vec_angle(np.array([[0,0,1]]).T, vec_legs)
    CFG['PJ1flexion'] = tempangle + tempangle2 + CFG['somersalt']
    # right leg
    tempangle,CFG['K1K2flexion'] =\
        calc_two_link_angles(H.K1.length, dk2, DK)
    CFG['PK1flexion'] = tempangle + tempangle2 + CFG['somersalt']

    # arms second. only somersalt can be specified, other torso
    # configuration variables must be zero

    def dist_hand_handle(angles, r_sh_hb, r_sh_h):
        """
        Returns the norm of the difference of the vector from the shoulder to
        the hand to the vector from the shoulder to the handlebar (i.e. the
        distance from the hand to the handlebar).

        Parameters
        ----------
        angles : array_like, shape(2,)
            The first angle is the elevation angle of the arm and the second is
            the abduction angle, both relative to the chest using euler 1-2-3
            angles.
        r_sh_hb : numpy.matrix, shape(3,1)
            The vector from the shoulder to the handlebar expressed in the
            chest reference frame.
        r_sh_h : numpy.matrix, shape(3,1)
            The vector from the shoulder to the hand (elbow is bent) expressed
            in the arm reference frame.

        Returns
        -------
        distance : float
            The distance from the handlebar point to the hand.

        """

        # these are euler rotation functions
        def x_rot(angle):
            sa = np.sin(angle)
            ca = np.cos(angle)
            Rx = np.matrix([[1., 0. , 0.],
                            [0., ca, sa],
                            [0., -sa, ca]])
            return Rx

        def y_rot(angle):
            sa = np.sin(angle)
            ca = np.cos(angle)
            Ry = np.matrix([[ca, 0. , -sa],
                            [0., 1., 0.],
                            [sa, 0., ca]])
            return Ry

        elevation = angles[0]
        abduction = angles[1]

        # create the rotation matrix of A (arm) in C (chest)
        R_A_C = y_rot(abduction) * x_rot(elevation)

        # express the vector from the shoulder to the hand in the C (chest)
        # refernce frame
        r_sh_h = R_A_C.T * r_sh_h

        return np.linalg.norm(r_sh_h - r_sh_hb)

    # left arm
    ##########
    tempangle, CFG['A1A2flexion'] =\
        calc_two_link_angles(H.A1.length, da2, DA)

    # this is the angle between the vector from the seat to the shoulder center
    # and the vector from the shoulder center to the handlebar center
    tempangle2 = vec_angle(vec_project(H.A1.pos - pos_seat, 1),
                            vec_project(pos_handl - H.A1.pos, 1))

    # the somersault angle plus the angle between the z unit vector and the
    # vector from the left shoulder to the left hand
    tempangle2 = CFG['somersalt'] + vec_angle(np.array([[0, 0, 1]]).T,
                                              pos_handl - H.A1.pos)

    # subtract the angle due to the arm not being straight
    CFG['CA1elevation'] = tempangle2 - tempangle

    # the angle between the vector from the shoulder to the handlebar and its
    # projection in the sagittal plane
    CFG['CA1abduction'] = vec_angle(pos_handl - H.A1.pos,
                                    vec_project(pos_handl - H.A1.pos, 1))

    # vector from the left shoulder to the left handlebar expressed in the
    # benchmark coordinates
    r_sh_hb = pos_handl - H.A1.pos
    # express r_sh_hb in the chest frame
    R_C_N = H.C.RotMat.T # transpose because Chris defined opposite my convention
    r_sh_hb = R_C_N * r_sh_hb
    # vector from the left shoulder to the hand (elbow bent) expressed in the
    # chest frame
    r_sh_h = np.mat([[0.],
                     [-da2 * np.sin(CFG['A1A2flexion'])],
                     [(-(H.A1.length + da2 *
                      np.cos(CFG['A1A2flexion'])))]])

    # chris defines his rotations relative to the arm coordinates but the
    # dist_hand_handle function is relative to the chest coordinates, thus the
    # negative guess
    guess = np.array([-CFG['CA1elevation'], -CFG['CA1abduction']])
    # home in on the exact solution
    elevation, abduction = fmin(dist_hand_handle, guess,
                                args=(r_sh_hb, r_sh_h), disp=False)
    # set the angles
    CFG['CA1elevation'], CFG['CA1abduction'] = -elevation, -abduction

    # right arm
    ###########
    tempangle, CFG['B1B2flexion'] =\
        calc_two_link_angles(H.B1.length, db2, DB)

    tempangle2 = vec_angle(vec_project(H.B1.pos - pos_seat, 1),
                           vec_project(pos_handr - H.B1.pos, 1))

    tempangle2 = CFG['somersalt'] + vec_angle(np.array([[0,0,1]]).T,
                                              pos_handr - H.B1.pos)

    CFG['CB1elevation'] = tempangle2 - tempangle
    CFG['CB1abduction'] = vec_angle(pos_handr - H.B1.pos,
                                    vec_project(pos_handr - H.B1.pos, 1))

    # vector from the left shoulder to the left handlebar expressed in the
    # benchmark coordinates
    r_sh_hb = pos_handr - H.B1.pos
    # express r_sh_hb in the chest frame
    R_C_N = H.C.RotMat.T # transpose because Chris defined opposite my convention
    r_sh_hb = R_C_N * r_sh_hb
    # vector from the left shoulder to the hand (elbow bent) expressed in the
    # chest frame
    r_sh_h = np.mat([[0.],
                     [-db2 * np.sin(CFG['B1B2flexion'])],
                     [(-(H.B1.length + db2 *
                      np.cos(CFG['B1B2flexion'])))]])

    # chris defines his rotations relative to the arm coordinates but the
    # dist_hand_handle function is relative to the chest coordinates, thus the
    # one negative guess
    guess = np.array([-CFG['CB1elevation'], CFG['CB1abduction']])
    # home in on the exact solution
    elevation, abduction = fmin(dist_hand_handle, guess,
                                args=(r_sh_hb, r_sh_h), disp=False)
    # set the angles
    CFG['CB1elevation'], CFG['CB1abduction'] = -elevation, abduction

    # assign configuration to human and check that the solution worked
    H.set_CFG_dict(CFG)
    if (np.round(H.j[7].pos, 3) != np.round(pos_footl, 3)).any():
        print "Left leg's actual position does not match its desired " \
              "position near the bike's bottom bracket. Left leg actual " \
              "position:\n", H.j[7].pos ,".\nLeft leg desired position:\n",\
              pos_footl, ".\nLeft leg base to end distance:", \
              np.linalg.norm(H.j[7].pos - H.J1.pos), ", Left leg D:", DJ
    if (np.round(H.k[7].pos, 3) != np.round(pos_footr, 3)).any():
        print "Right leg's actual position does not match its desired " \
              "position near the bike's bottom bracket. Right leg actual " \
              "position:\n", H.k[7].pos, ".\nRight leg desired position:\n",\
              pos_footr, ".\nRight leg base to end distance:", \
              np.linalg.norm(H.k[7].pos - H.K1.pos), ", Left leg D:", DK
    if (np.round(H.a[6].pos, 3) != np.round(pos_handl, 3)).any():
        print "Left arm's actual position does not match its desired " \
              "position on the bike's handlebar. Left arm actual " \
              "position:\n",H.A2.endpos,".\nLeft arm desired position:\n",\
              pos_handl, "\nLeft arm base to end distance:", \
              np.linalg.norm(H.A2.endpos - H.A1.pos),", Left arm D:", DA
    if (np.round(H.b[6].pos, 3) != np.round(pos_handr, 3)).any():
        print "Right arm's actual position does not match its desired " \
              "position on the bike's handrebar. Right arm actual " \
              "position:", H.B2.endpos ,".\nRight arm desired position:\n",\
              pos_handr, ".\nRight arm base to end distance:", \
              np.linalg.norm(H.B2.endpos - H.B1.pos), ", Right arm D:", DB

    # draw rider for fun, but possibly to check results aren't crazy
    if drawrider==True:
        H.draw_visual(forward=(0,-1,0),up=(0,0,-1))
        H.draw_vector('origin',pos_footl)
        H.draw_vector('origin',pos_footr)
        H.draw_vector('origin',pos_handl)
        H.draw_vector('origin',pos_handr)
        H.draw_vector('origin',H.A2.endpos)
        H.draw_vector('origin',H.A2.endpos,(0,0,1))
        H.draw_vector('origin',H.B2.endpos,(0,0,1))
    return H

def combine_bike_rider(bicyclePar, riderPar):
    """
    Combines the inertia of the bicycle frame with the
    inertia of a rider.

    Parameters
    ----------
    bicyclePar : dictionary
        The benchmark parameter set of a bicycle.
    riderPar : dictionary
        The rider's mass, center of mass, and inertia expressed in the
        benchmark bicycle reference frame.

    Returns
    -------
    bicyclePar : dictionary
        The benchmark bicycle parameters with a rigid rider added to the
        bicycle frame.

    """

    # list the masses of the rider and bicycle
    masses = np.array([riderPar['mB'], bicyclePar['mB']])
    # list the centers of mass of the rider and bicycle
    coordinates = np.array([[riderPar['xB'], bicyclePar['xB']],
                            [riderPar['yB'], 0.],
                            [riderPar['zB'], bicyclePar['zB']]])
    # calculate the new mass and center of mass
    mT, cT = total_com(coordinates, masses)
    # get inertia tensors for the bicycle and rider
    IRider = part_inertia_tensor(riderPar, 'B')
    IBicycle = part_inertia_tensor(bicyclePar, 'B')
    # calculate the distance from the center of mass of each body to the
    # center of mass of the combined body
    dRider = np.array([riderPar['xB'] - cT[0],
                       riderPar['yB'] - cT[1],
                       riderPar['zB'] - cT[2]])
    dBicycle = np.array([bicyclePar['xB'] - cT[0],
                         0.,
                         bicyclePar['zB'] - cT[2]])
    # calculate the total inertia about the total body center of mass
    I = (parallel_axis(IRider, riderPar['mB'], dRider) +
         parallel_axis(IBicycle, bicyclePar['mB'], dBicycle))
    # assign new inertia back to bike
    bicyclePar['xB'] = cT[0]
    bicyclePar['zB'] = cT[2]
    bicyclePar['yB'] = 0.0
    bicyclePar['mB'] = mT
    bicyclePar['IBxx'] = I[0, 0]
    bicyclePar['IBxz'] = I[0, 2]
    bicyclePar['IByy'] = I[1, 1]
    bicyclePar['IBzz'] = I[2, 2]

    return bicyclePar

def point_to_line_distance(point, pointsOnLine):
    '''Returns the minimal distance from a point to a line in three
    dimensional space.

    Parameters
    ----------
    point : ndarray, shape(3,)
        The x, y, and z coordinates of a point.
    pointsOnLine : ndarray, shape(3,2)
        The x, y, and z coordinates of two points on a line. Rows are
        coordinates and columns are points.

    Returns
    -------
    distance : float
        The minimal distance from the line to the point.

    '''
    x1 = pointsOnLine[:, 0]
    x2 = pointsOnLine[:, 1]
    x0 = point

    def norm(v):
        return unumpy.sqrt(np.dot(v, v))

    distance = norm(np.cross((x0 - x1), (x0 - x2))) / norm(x2 - x1)

    return distance

def plot_eigenvalues(bikes, speeds, colors=None, linestyles=None, largest=False):
    '''Returns a figure with the eigenvalues vs speed for multiple bicycles.

    Parameters
    ----------
    bikes : list
        A list of Bicycle objects.
    speeds : ndarray, shape(n,)
        An array of speeds.
    colors : list
        A list of matplotlib colors for each bicycle.
    linestyles : list
        A list of matplotlib linestyles for each bicycle.
    largest : boolean
        If true, only plots the largest eigenvalue.

    Returns
    -------
    fig : matplotlib figure

    '''

    numBikes = len(bikes)

    if not colors:
        # define some colors for the parts
        cmap = plt.get_cmap('gist_rainbow')
        colors = []
        for i in range(numBikes):
            colors.append(cmap(1. * i / numBikes))

    if not linestyles:
        linestyles = ['-'] * numBikes

    for i, bike in enumerate(bikes):
        if i == 0:
            fig = None
        fig = bike.plot_eigenvalues_vs_speed(speeds, fig=fig, color=colors[i],
                                             largest=largest, generic=True,
                                             linestyle=linestyles[i],
                                             show=False)
        plt.legend()
        plt.axis('tight')

    plt.show()

    return fig

def project_point_on_line(line, point):
    '''Returns point of projection.

    Parameters
    ----------
    line : tuple
        Slope and intercept of the line.
    point : tuple
        Location of the point.

    Returns
    -------
    newPoint : tuple
        The location of the projected point.

    '''
    m, b = line
    c , d = point
    x = (m * d + c - m * b) / (m**2. + 1.)
    y = (m**2. * d + m * c + b) / (m**2. + 1.)
    return x, y

def get_parts_in_parameters(par):
    '''Returns a list of parts in a parameter dictionary.

    Parameters
    ----------
    par : dictionary
        Benchmark bicycle parameters.

    Returns
    -------
    parts : list
        Unique list of parts that contain one or more of 'H', 'B', 'F', 'R',
        'S', 'G'.

    '''
    parts = [x[1] for x in par.keys() if x.startswith('m')]
    return parts

def write_parameter_text_file(pathToTxtFile, parDict):
    '''Writes parameter set to file.

    Parameters
    ----------
    pathToTxtFile : string
        The path to the file to write the parameters.
    pardict : dictionary
        A dictionary of parameters for the bicycle.

    Returns
    -------
    saved : boolean
        True if the file was saved and false if not.

    '''

    # make the Parameters directory if it doesn't exist
    head, tail = os.path.split(pathToTxtFile)
    if not os.path.isdir(head):
        print "Created direcotry %s" % head
        os.makedirs(head)

    try:
        f = open(pathToTxtFile)
        f.close()
        del f
        ans = None
        while ans !=  'y' and ans != 'n':
            ans = raw_input("%s exists already. Are you sure you want" \
                            " to overwrite it? (y or n)\n" % pathToTxtFile)
        if ans == 'y':
            f = open(pathToTxtFile, 'w')
    except IOError:
        f = open(pathToTxtFile, 'w')

    try:
        f
        keys = sorted(parDict.keys())
        for key in keys:
            f.write(key + ' = ' + str(parDict[key]) + '\n')
        f.close()
        print "Parameters saved to %s" % pathToTxtFile
        return True
    except UnboundLocalError:
        print "%s was not saved." % pathToTxtFile
        return False

def sort_modes(evals, evecs):
    '''Sort eigenvalues and eigenvectors into weave, capsize, caster modes.

    Parameters
    ----------
    evals : ndarray, shape (n, 4)
        eigenvalues
    evecs : ndarray, shape (n, 4, 4)
        eigenvectors

    Returns
    -------
    weave['evals'] : ndarray, shape (n, 2)
        The eigen value pair associated with the weave mode.
    weave['evecs'] : ndarray, shape (n, 4, 2)
        The associated eigenvectors of the weave mode.
    capsize['evals'] : ndarray, shape (n,)
        The real eigenvalue associated with the capsize mode.
    capsize['evecs'] : ndarray, shape(n, 4, 1)
        The associated eigenvectors of the capsize mode.
    caster['evals'] : ndarray, shape (n,)
        The real eigenvalue associated with the caster mode.
    caster['evecs'] : ndarray, shape(n, 4, 1)
        The associated eigenvectors of the caster mode.

    Notes
    -----
    This only works on the standard bicycle eigenvalues, not necessarily on any
    general eigenvalues for the bike model (e.g. there isn't always a distinct weave,
    capsize and caster). Some type of check unsing the derivative of the curves
    could make it more robust.

    '''
    evalsorg = np.zeros_like(evals)
    evecsorg = np.zeros_like(evecs)
    # set the first row to be the same
    evalsorg[0] = evals[0]
    evecsorg[0] = evecs[0]
    # for each speed
    for i, speed in enumerate(evals):
        if i == evals.shape[0] - 1:
            break
        # for each current eigenvalue
        used = []
        for j, e in enumerate(speed):
            try:
                x, y = np.real(evalsorg[i, j].nominal_value), np.imag(evalsorg[i, j].nominal_value)
            except:
                x, y = np.real(evalsorg[i, j]), np.imag(evalsorg[i, j])
            # for each eigenvalue at the next speed
            dist = np.zeros(4)
            for k, eignext in enumerate(evals[i + 1]):
                try:
                    xn, yn = np.real(eignext.nominal_value), np.imag(eignext.nominal_value)
                except:
                    xn, yn = np.real(eignext), np.imag(eignext)
                # distance between points in the real/imag plane
                dist[k] = np.abs(((xn - x)**2 + (yn - y)**2)**0.5)
            if np.argmin(dist) in used:
                # set the already used indice higher
                dist[np.argmin(dist)] = np.max(dist) + 1.
            else:
                pass
            evalsorg[i + 1, j] = evals[i + 1, np.argmin(dist)]
            evecsorg[i + 1, :, j] = evecs[i + 1, :, np.argmin(dist)]
            # keep track of the indices we've used
            used.append(np.argmin(dist))
    weave = {'evals' : evalsorg[:, 2:], 'evecs' : evecsorg[:, :, 2:]}
    capsize = {'evals' : evalsorg[:, 1], 'evecs' : evecsorg[:, :, 1]}
    caster = {'evals' : evalsorg[:, 0], 'evecs' : evecsorg[:, :, 0]}
    return weave, capsize, caster

def remove_uncertainties(dictionary):
    '''Returns a dictionary with the uncertainties removed.'''
    noUncert = {}
    for k, v in dictionary.items():
        try:
            # this is the case if the value is a single uncertainty
            noUncert[k] = v.nominal_value
        except AttributeError:
            # this is the case if the value is an array of ufloats
            try:
                noUncert[k] = [x.nominal_value for x in v]
            except TypeError:
                # this is the case if the value is a float
                noUncert[k] = v
    return noUncert

def benchmark_par_to_canonical(p):
    '''Returns the canonical matrices of the Whipple bicycle model linearized
    about the upright constant velocity configuration. It uses the parameter
    definitions from Meijaard et al. 2007.

    Parameters
    ----------
    p : dictionary
        A dictionary of the benchmark bicycle parameters. Make sure your units
        are correct, best to ue the benchmark paper's units!

    Returns
    -------
    M : ndarray, shape(2,2)
        The mass matrix.
    C1 : ndarray, shape(2,2)
        The damping like matrix that is proportional to the speed, v.
    K0 : ndarray, shape(2,2)
        The stiffness matrix proportional to gravity, g.
    K2 : ndarray, shape(2,2)
        The stiffness matrix proportional to the speed squared, v**2.

    Notes
    -----
    This function handles parameters with uncertanties.

    '''
    mT = p['mR'] + p['mB'] + p['mH'] + p['mF']
    xT = (p['xB'] * p['mB'] + p['xH'] * p['mH'] + p['w'] * p['mF']) / mT
    zT = (-p['rR'] * p['mR'] + p['zB'] * p['mB'] +
          p['zH'] * p['mH'] - p['rF'] * p['mF']) / mT

    ITxx = (p['IRxx'] + p['IBxx'] + p['IHxx'] + p['IFxx'] + p['mR'] *
            p['rR']**2 + p['mB'] * p['zB']**2 + p['mH'] * p['zH']**2 + p['mF']
            * p['rF']**2)
    ITxz = (p['IBxz'] + p['IHxz'] - p['mB'] * p['xB'] * p['zB'] -
            p['mH'] * p['xH'] * p['zH'] + p['mF'] * p['w'] * p['rF'])
    p['IRzz'] = p['IRxx']
    p['IFzz'] = p['IFxx']
    ITzz = (p['IRzz'] + p['IBzz'] + p['IHzz'] + p['IFzz'] +
            p['mB'] * p['xB']**2 + p['mH'] * p['xH']**2 + p['mF'] * p['w']**2)

    mA = p['mH'] + p['mF']
    xA = (p['xH'] * p['mH'] + p['w'] * p['mF']) / mA
    zA = (p['zH'] * p['mH'] - p['rF']* p['mF']) / mA

    IAxx = (p['IHxx'] + p['IFxx'] + p['mH'] * (p['zH'] - zA)**2 +
            p['mF'] * (p['rF'] + zA)**2)
    IAxz = (p['IHxz'] - p['mH'] * (p['xH'] - xA) * (p['zH'] - zA) + p['mF'] *
            (p['w'] - xA) * (p['rF'] + zA))
    IAzz = (p['IHzz'] + p['IFzz'] + p['mH'] * (p['xH'] - xA)**2 + p['mF'] *
            (p['w'] - xA)**2)
    uA = (xA - p['w'] - p['c']) * umath.cos(p['lam']) - zA * umath.sin(p['lam'])
    IAll = (mA * uA**2 + IAxx * umath.sin(p['lam'])**2 +
            2 * IAxz * umath.sin(p['lam']) * umath.cos(p['lam']) +
            IAzz * umath.cos(p['lam'])**2)
    IAlx = (-mA * uA * zA + IAxx * umath.sin(p['lam']) + IAxz *
            umath.cos(p['lam']))
    IAlz = (mA * uA * xA + IAxz * umath.sin(p['lam']) + IAzz *
            umath.cos(p['lam']))

    mu = p['c'] / p['w'] * umath.cos(p['lam'])

    SR = p['IRyy'] / p['rR']
    SF = p['IFyy'] / p['rF']
    ST = SR + SF
    SA = mA * uA + mu * mT * xT

    Mpp = ITxx
    Mpd = IAlx + mu * ITxz
    Mdp = Mpd
    Mdd = IAll + 2 * mu * IAlz + mu**2 * ITzz
    M = np.array([[Mpp, Mpd], [Mdp, Mdd]])

    K0pp = mT * zT # this value only reports to 13 digit precision it seems?
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA * umath.sin(p['lam'])
    K0 = np.array([[K0pp, K0pd], [K0dp, K0dd]])

    K2pp = 0.
    K2pd = (ST - mT * zT) / p['w'] * umath.cos(p['lam'])
    K2dp = 0.
    K2dd = (SA + SF * umath.sin(p['lam'])) / p['w'] * umath.cos(p['lam'])
    K2 = np.array([[K2pp, K2pd], [K2dp, K2dd]])

    C1pp = 0.
    C1pd = (mu*ST + SF*umath.cos(p['lam']) + ITxz / p['w'] *
            umath.cos(p['lam']) - mu*mT*zT)
    C1dp = -(mu * ST + SF * umath.cos(p['lam']))
    C1dd = (IAlz / p['w'] * umath.cos(p['lam']) + mu * (SA +
            ITzz / p['w'] * umath.cos(p['lam'])))
    C1 = np.array([[C1pp, C1pd], [C1dp, C1dd]])

    return M, C1, K0, K2

def abMatrix(M, C1, K0, K2, v, g):
    '''Calculate the A and B matrices for the Whipple bicycle model linearized
    about the upright configuration.

    Parameters
    ----------
    M : ndarray, shape(2,2)
        The mass matrix.
    C1 : ndarray, shape(2,2)
        The damping like matrix that is proportional to the speed, v.
    K0 : ndarray, shape(2,2)
        The stiffness matrix proportional to gravity, g.
    K2 : ndarray, shape(2,2)
        The stiffness matrix proportional to the speed squared, v**2.
    v : float
        Forward speed.
    g : float
        Acceleration due to gravity.

    Returns
    -------
    A : ndarray, shape(4,4)
        State matrix.
    B : ndarray, shape(4,2)
        Input matrix.

    The states are [roll rate,
                    steer rate,
                    roll angle,
                    steer angle]
    The inputs are [roll torque,
                    steer torque]

    '''

    a11 = -v * C1
    a12 = -(g * K0 + v**2 * K2)
    a21 = np.eye(2)
    a22 = np.zeros((2, 2))
    A = np.vstack((np.dot(unumpy.ulinalg.inv(M), np.hstack((a11, a12))),
                   np.hstack((a21, a22))))
    B = np.vstack((unumpy.ulinalg.inv(M), np.zeros((2, 2))))

    return A, B

def part_inertia_tensor(par, part):
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
    if isinstance(par['mB'], UFloat):
        dtype=object
    else:
        dtype='float64'
    I = np.zeros((3, 3), dtype=dtype)
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
    d1 = np.cos(par['lam']) * (par['c'] + par['w'] -
                par['rR'] * np.tan(par['lam']))
    d3 = -np.cos(par['lam']) * (par['c'] - par['rF'] *
                np.tan(par['lam']))
    x = np.zeros(4, dtype=object)
    z = np.zeros(4, dtype=object)
    x[0] = 0.
    x[1] = d1 * np.cos(par['lam'])
    x[2] = par['w'] - d3 * np.cos(par['lam'])
    x[3] = par['w']
    z[0] = -par['rR']
    z[1] = -par['rR'] - d1 * np.sin(par['lam'])
    z[2] = -par['rF'] + d3 * np.sin(par['lam'])
    z[3] = -par['rF']

    return x, z

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


    # local accelation due to gravity
    par['g'] = mp['g']

    # calculate the wheel y inertias
    par['IFyy'] = compound_pendulum_inertia(mp['mF'], mp['g'],
                                            mp['lF'], mp['TcF1'])
    par['IRyy'] = compound_pendulum_inertia(mp['mR'], mp['g'],
                                            mp['lR'], mp['TcR1'])

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

    # calculate the stiffness of the torsional pendulum
    IPxx, IPyy, IPzz = tube_inertia(mp['lP'], mp['mP'], mp['dP'] / 2., 0.)
    torStiff = torsional_pendulum_stiffness(IPyy, mp['TtP1'])
    #print "Torsional pendulum stiffness:", torStiff

    # calculate the wheel x/z inertias
    par['IFxx'] = tor_inertia(torStiff, mp['TtF1'])
    par['IRxx'] = tor_inertia(torStiff, mp['TtR1'])

    pendulumInertias = {}

    # calculate the in plane moments of inertia
    for part, slopeSet in slopes.items():
        # the number of orientations for this part
        numOrien = len(slopeSet)
        # intialize arrays to store the inertia values and orientation angles
        penInertia = np.zeros(numOrien, dtype=object)
        beta = np.array(betas[part])
        # fill arrays of the inertias
        for i in range(numOrien):
            penInertia[i] = tor_inertia(torStiff, mp['Tt' + part + str(i + 1)])
        # store these inertias
        pendulumInertias[part] = list(penInertia)
        inertia = inertia_components(penInertia, beta)
        for i, axis in enumerate(['xx', 'xz', 'zz']):
            par['I' + part + axis] = inertia[i]

    if forkIsSplit:
        # combine the moments of inertia to find the total handlebar/fork MoI
        IG = part_inertia_tensor(par, 'G')
        IS = part_inertia_tensor(par, 'S')
        # columns are parts, rows = x, y, z
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

    # package the extra information that is useful outside this function
    extras = {'slopes' : slopes,
              'intercepts' : intercepts,
              'betas' : betas,
              'pendulumInertias' : pendulumInertias}

    return par, extras

def torsional_pendulum_stiffness(I, T):
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
    the keyword is either 'B', 'H', 'G' or 'S' for Frame, Handlebar/Fork,
    Handlerbar, and Fork respectively.

    '''
    # find the slope and intercept for pendulum axis
    if forkIsSplit:
        l1, l2 = calculate_l1_l2(mp['h6'], mp['h7'], mp['d5'],
                                 mp['d6'], mp['l'])
        slopes = {'B':[], 'G':[], 'S':[]}
        intercepts = {'B':[], 'G':[], 'S':[]}
        betas = {'B':[], 'G':[], 'S':[]}
    else:
        l1, l2 = 0., 0.
        slopes = {'B':[], 'H':[]}
        intercepts = {'B':[], 'H':[]}
        betas = {'B':[], 'H':[]}

    # get the alpha keys and put them in order
    listOfAlphaKeys = [x for x in mp.keys() if x.startswith('alpha')]
    listOfAlphaKeys.sort()

    # caluclate m, b and beta for each orientation
    for key in listOfAlphaKeys:
        alpha = mp[key]
        a = mp['a' + key[5:]]
        part = key[5]
        m, b, beta = com_line(alpha, a, par, part, l1, l2)
        slopes[part].append(m)
        intercepts[part].append(b)
        betas[part].append(beta)

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
        print part, "doesn't exist"
        raise KeyError

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

    Parameters
    ----------
    rF: float
        The front wheel radius
    lam: float
        The steer axis tilt (pi/2 - headtube angle). The angle between the
        headtube and a vertical line.
    fo: float
        The fork offset

    Returns
    -------
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

    # The following assumes that the uncertainty calculated for the guess is
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

def fit_goodness(ym, yp):
    '''
    Calculate the goodness of fit.

    Parameters
    ----------
    ym : vector of measured values
    yp : vector of predicted values

    Returns
    -------
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
    """
    Returns a dictionary of float and/or ufloat parameters from a parameter file.

    Parameters
    ----------
    pathToFile : string
        The path to the text file with the parameters listed in the specified
        format.

    Returns
    -------
    parameters : dictionary
        A dictionary of the values stored in the text files.

    For example::

        c = 0.08 +/- 0.01
        d=0.314+/-0.002
        t = 0.1+/-0.01, 0.12+/-0.02
        whb = 0.5

    The first item on the line must be the variable name and the second is an
    equals sign. The values to the right of the equal sign much may or may not
    contain an uncertainty designated by `+/-`. Multiple comma seperated values
    will be averaged.

    """

    parameters = {}
    # parse the text file
    with open(pathToFile, 'r') as f:
        for line in f:
            # ignore lines that start with a hash
            if line[0] != '#':
                # remove any whitespace characters and comments at the end of
                # the line, then split the right and left side of the equality
                equality = line.strip().split('#')[0].split('=')
                # ['a ', ' 0.1 +/- 0.05 , 0.09 +/- 0.05']
                valList = equality[1].strip().split(',')
                # ['0.1 +/- 0.05 ', ' 0.09 +/- 0.05']
                if '+/-' in equality[1]:
                    values = [ufloat(x) for x in valList]
                else:
                    values = [float(x) for x in valList]
                # store in dictionary
                parameters[equality[0].strip()] = np.mean(values)

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

def calc_two_link_angles(L1, L2, D):
    '''Solves a simple case of the two-link revolute joint inverse
    kinematics problem. Both output angles are positive. The simple case
    is taht the end of the second link lies on the x-axis.

    Parameters
    ----------
    L1 : float
        Length of the first link.
    L2 : float
        Length of the second link.
    D : float
        Distance from the base of first link to the end of the second link.

    Returns
    -------
    theta1 : float
        (radians) Angle between x-axis and first link; always positive.
    theta2 : float
        (radians) Angle between first link and second link; always positive.

    '''

    theta1 = np.arccos( (L1**2 + D**2 - L2**2) / (2.0 * L1 * D) )
    theta2 = theta1 + np.arcsin( L1 / L2 * np.sin( theta1 ) )

    return theta1, theta2

def vec_project(vec, direction):
    '''Vector projection into a plane, where the plane is defined by a
    normal vector.

    Parameters
    ----------
    vec : np.array(3,1)
        vector to be projected into a plane
    direction : int or np.array, shape(3,)
        If int, it is one of the three orthogonal directions, (0,1 or 2) of
        the input vector (essentially, that component of vec is set to zero).
        If np.array, can be in any direction (not necessarily a coordinate
        direction).

    Returns
    -------
    vec_out : np.array(3,1)
        Projected vector.

    '''
    vec = vec.flatten()
    if type(direction) == int:
        unitdir = np.zeros(3)
        unitdir[direction] = 1
    elif type(direction) == np.array:
        unitdir = direction / np.linalg.norm(direction)
    proj = vec - np.dot(vec, unitdir)
    return proj.reshape((3, 1))

def vec_angle(v1,v2):
    '''Returns the interior angle between two vectors using the dot product. Inputs do not need to be unit vectors.

    Parameters
    ----------
    v1 : np.array (3,1)
        input vector.
    v2 : np.array (3,1)
        input vector.

    Returns
    -------
    angle : float
        (radians) interior angle between v1 and v2.
    '''
    return np.arccos( float(np.dot(v1.T,v2)) / (
           np.linalg.norm(v1) * np.linalg.norm(v2) ) )
