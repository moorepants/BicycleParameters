#!/usr/bin/env python

# builtin modules
import os

# dependencies
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge
from uncertainties import umath, unumpy

# local module imports
import bicycle
import inertia
import com
import io
import geometry
import period
import rider
#from plot import plot_eigenvalues

class Bicycle(object):
    """
    An object for a bicycle. A bicycle has parameters and can have a rider
    attached to it. That's about it for now.

    """

    def __new__(cls, bicycleName, pathToData='.', forceRawCalc=False,
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

    def __init__(self, bicycleName, pathToData='.', forceRawCalc=False,
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
                bike, ptype = io.space_out_camel_case(fname, output='list')
                # load the parameters
                pathToFile = os.path.join(parDir, parFile)
                self.parameters[ptype] = io.load_parameter_text_file(pathToFile)

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
                io.write_parameter_text_file(pathToTxtFile, parameters[pset])
                if self.hasRider:
                    pathToCombFile = os.path.join(pathToCombDir, fileName)
                    io.write_parameter_text_file(pathToCombFile,
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
                I = inertia.part_inertia_tensor(par, 'H')
                m = par['mH']
                x = par['xH']
                z = par['zH']
            elif fork and not handlebar:
                # fork alone
                I = inertia.part_inertia_tensor(par, 'S')
                m = par['mS']
                x = par['xS']
                z = par['zS']
            elif handlebar and not fork:
                # handlebar alone
                I = inertia.part_inertia_tensor(par, 'G')
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
            mAss, cAss = com.total_com(coords, masses)

            # front wheel inertia in the benchmark reference frame about the
            # com
            IF = inertia.part_inertia_tensor(par, 'F')

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
            iAss = (inertia.parallel_axis(I, m, d) +
                    inertia.parallel_axis(IF, par['mF'], dF))

            # this is the inertia of the assembly about a reference frame aligned with
            # the steer axis and through the center of mass
            iAssRot = inertia.rotate_inertia_tensor(iAss, par['lam'])

        else: # don't add the wheel
            mAss = m
            cAss = np.array([x, 0., z])
            iAssRot = inertia.rotate_inertia_tensor(I, par['lam'])

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
            distance = geometry.point_to_line_distance(cAss, pointsOnLine)
            print "handlebar cg distance", distance

            # now calculate the inertia about the steer axis of the rotated frame
            iAss = inertia.parallel_axis(iAssRot, mAss, np.array([distance, 0., 0.]))
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
        self.parameters['Measured'] = io.load_parameter_text_file(pathToRawFile)

        forkIsSplit = is_fork_split(self.parameters['Measured'])

        # if the the user doesn't specifiy to force period calculation, then
        # see if enough data is actually available in the *Measured.txt file to
        # do the calculations
        if not forcePeriodCalc:
            forcePeriodCalc = period.check_for_period(self.parameters['Measured'],
                                               forkIsSplit)

        if forcePeriodCalc == True:
            # get the list of mat files associated with this bike
            matFiles = [x for x in os.listdir(rawDataDir)
                        if x.endswith('.mat')]
            matFiles.sort()
            # calculate the period for each file for this bicycle
            periods = period.calc_periods_for_files(rawDataDir, matFiles, forkIsSplit)
            # add the periods to the measured parameters
            self.parameters['Measured'].update(periods)

            io.write_periods_to_file(pathToRawFile, periods)

        return calculate_benchmark_from_measured(self.parameters['Measured'])

    def add_rider(self, riderName, reCalc=False, draw=False):
        """
        Adds the inertial effects of a rigid rider to the bicycle.

        Parameters
        ----------
        riderName : string
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
                  "{0}.".format(riderName))
            pathToData = os.path.split(os.path.split(self.directory)[0])[0]
            # get the path to the rider's folder
            pathToRider = os.path.join(pathToData, 'riders', riderName)
            # load in the parameters
            bicyclePar = self.parameters['Benchmark']
            bicycleName = self.bicycleName

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
                    rider.configure_rider(pathToRider, bicycleName, bicyclePar,
                            measuredPar, draw)
            else:
                pathToParFile = os.path.join(pathToRider, 'Parameters',
                    riderName + self.bicycleName + 'Benchmark.txt')
                try:
                    # load the parameter file
                    riderPar = io.load_parameter_text_file(pathToParFile)
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
                        rider.configure_rider(pathToRider, bicycleName,
                                bicyclePar, measuredPar, draw)
                else:
                    print("Loaded the precalculated parameters from " +
                          "{0}".format(pathToParFile))
                    bicycleRiderPar = inertia.combine_bike_rider(bicyclePar, riderPar)
            # set the attributes
            self.riderPar['Benchmark'] = riderPar
            try:
                self.human = human
            except NameError:
                self.human = None
            self.parameters['Benchmark'] = bicycleRiderPar
            self.riderName = riderName
            self.hasRider = True

    def plot_bicycle_geometry(self, show=True, pendulum=True,
                              centerOfMass=True, inertiaEllipse=True):
        '''Returns a figure showing the basic bicycle geometry, the centers of
        mass and the moments of inertia.

        '''
        par = io.remove_uncertainties(self.parameters['Benchmark'])
        parts = get_parts_in_parameters(par)

        try:
            slopes = io.remove_uncertainties(self.extras['slopes'])
            intercepts = io.remove_uncertainties(self.extras['intercepts'])
            penInertias = io.remove_uncertainties(self.extras['pendulumInertias'])
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
                I = inertia.part_inertia_tensor(par, part)
                Ip, C = inertia.principal_axes(I)
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
        deex, deez = geometry.fundamental_geometry_plot_data(par)
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
                    xPoint, zPoint = geometry.project_point_on_line((m, b),
                            (xcom, zcom))
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

        M, C1, K0, K2 = bicycle.benchmark_par_to_canonical(par)

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

        A, B = bicycle.ab_matrix(M, C1, K0, K2, speed, g)

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

        par = io.remove_uncertainties(self.parameters['Benchmark'])

        M, C1, K0, K2 = bicycle.benchmark_par_to_canonical(par)

        m, n = 4, speeds.shape[0]
        evals = np.zeros((n, m), dtype='complex128')
        evecs = np.zeros((n, m, m), dtype='complex128')
        for i, speed in enumerate(speeds):
            A, B = bicycle.ab_matrix(M, C1, K0, K2, speed, par['g'])
            w, v = np.linalg.eig(A)
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
            wea, cap, cas = bicycle.sort_modes(evals, evecs)

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
    par = geometry.calculate_benchmark_geometry(mp, par)

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
    slopes, intercepts, betas = com.part_com_lines(mp, par, forkIsSplit)

    # calculate the centers of mass
    for part in slopes.keys():
        par['x' + part], par['z' + part] = com.center_of_mass(slopes[part],
            intercepts[part])

    # find the center of mass of the handlebar/fork assembly if the fork was
    # split
    if forkIsSplit:
        coordinates = np.array([[par['xS'], par['xG']],
                                [0., 0.],
                                [par['zS'], par['zG']]])
        masses = np.array([par['mS'], par['mG']])
        mH, cH = inertia.total_com(coordinates, masses)
        par['mH'] = mH
        par['xH'] = cH[0]
        par['zH'] = cH[2]


    # local accelation due to gravity
    par['g'] = mp['g']

    # calculate the wheel y inertias
    par['IFyy'] = inertia.compound_pendulum_inertia(mp['mF'], mp['g'],
                                            mp['lF'], mp['TcF1'])
    par['IRyy'] = inertia.compound_pendulum_inertia(mp['mR'], mp['g'],
                                            mp['lR'], mp['TcR1'])

    # calculate the y inertias for the frame and fork
    lB = (par['xB']**2 + (par['zB'] + par['rR'])**2)**(0.5)
    par['IByy'] = inertia.compound_pendulum_inertia(mp['mB'], mp['g'], lB,
                                                    mp['TcB1'])

    if forkIsSplit:
        # fork
        lS = ((par['xS'] - par['w'])**2 +
              (par['zS'] + par['rF'])**2)**(0.5)
        par['ISyy'] = inertia.compound_pendulum_inertia(mp['mS'], mp['g'],
                                                lS, mp['TcS1'])
        # handlebar
        l1, l2 = geometry.calculate_l1_l2(mp['h6'], mp['h7'],
                                 mp['d5'], mp['d6'], mp['l'])
        u1, u2 = geometry.fwheel_to_handlebar_ref(par['lam'], l1, l2)
        lG = ((par['xG'] - par['w'] + u1)**2 +
              (par['zG'] + par['rF'] + u2)**2)**(.5)
        par['IGyy'] = inertia.compound_pendulum_inertia(mp['mG'], mp['g'],
                                                lG, mp['TcG1'])
    else:
        lH = ((par['xH'] - par['w'])**2 +
              (par['zH'] + par['rF'])**2)**(0.5)
        par['IHyy'] = inertia.compound_pendulum_inertia(mp['mH'], mp['g'],
                                                lH, mp['TcH1'])

    # calculate the stiffness of the torsional pendulum
    IPxx, IPyy, IPzz = inertia.tube_inertia(mp['lP'], mp['mP'],
                                            mp['dP'] / 2., 0.)
    torStiff = inertia.torsional_pendulum_stiffness(IPyy, mp['TtP1'])
    #print "Torsional pendulum stiffness:", torStiff

    # calculate the wheel x/z inertias
    par['IFxx'] = inertia.tor_inertia(torStiff, mp['TtF1'])
    par['IRxx'] = inertia.tor_inertia(torStiff, mp['TtR1'])

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
            penInertia[i] = inertia.tor_inertia(torStiff, mp['Tt' + part + str(i + 1)])
        # store these inertias
        pendulumInertias[part] = list(penInertia)
        inert = inertia.inertia_components(penInertia, beta)
        for i, axis in enumerate(['xx', 'xz', 'zz']):
            par['I' + part + axis] = inert[i]

    if forkIsSplit:
        # combine the moments of inertia to find the total handlebar/fork MoI
        IG = inertia.part_inertia_tensor(par, 'G')
        IS = inertia.part_inertia_tensor(par, 'S')
        # columns are parts, rows = x, y, z
        coordinates = np.array([[par['xG'], par['xS']],
                                [0., 0.],
                                [par['zG'], par['zS']]])
        masses = np.array([par['mG'], par['mS']])
        par['mH'], cH = com.total_com(coordinates, masses)
        par['xH'] = cH[0]
        par['zH'] = cH[2]
        dG = np.array([par['xG'] - par['xH'], 0., par['zG'] - par['zH']])
        dS = np.array([par['xS'] - par['xH'], 0., par['zS'] - par['zH']])
        IH = (inertia.parallel_axis(IG, par['mG'], dG) +
              inertia.parallel_axis(IS, par['mS'], dS))
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
