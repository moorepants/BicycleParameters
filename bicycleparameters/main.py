#!/usr/bin/env python

# builtin modules
import os

# dependencies
from dtk import control
from matplotlib.patches import Ellipse, Wedge
from uncertainties import unumpy
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly
except ImportError:
    px = None
    go = None
else:
    del plotly
    import plotly.express as px
    import plotly.graph_objects as go

# local module imports
from . import bicycle
from . import inertia
from . import com
from . import io
from . import geometry
from . import period
from . import rider
from . import plot

GOLDEN_RATIO = (1.0 + np.sqrt(5.0))/2.0


class Bicycle(object):
    """
    An object for a bicycle. A bicycle has parameters and can have a rider
    attached to it. That's about it for now.

    """

    def __new__(cls, bicycleName, pathToData='.', forceRawCalc=False,
                forcePeriodCalc=False):
        '''Returns a NoneType object if there is no directory for the
        bicycle.'''
        # is there a data directory for this bicycle? if not, tell the user to
        # put some data in the folder so we have something to work with!
        try:
            pathToBicycle = os.path.join(pathToData, 'bicycles', bicycleName)
            if os.path.isdir(pathToBicycle):
                print("We have foundeth a directory named: " +
                      "{0}.".format(pathToBicycle))
                return super(Bicycle, cls).__new__(cls)
            else:
                raise ValueError
        except:
            mes = """Are you nuts?! Make a directory called '{0}' with basic
data for your bicycle in this directory: '{1}'. Then I can actually create a
bicycle object. You may either need to change to the correct directory or reset
the pathToData argument.""".format(bicycleName, pathToData)
            print(mes)
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
            Forces a recalculation of the benchmark parameters from the
            measured parameters. Otherwise it will only run the calculation if
            there is no benchmark parameter file.
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
                self.parameters[ptype] = io.load_parameter_text_file(
                    pathToFile)

        # this is where the raw data files from the pendulum oscillations are
        # stored
        rawDataDir = os.path.join(self.directory, 'RawData')

        # it would be more robust to see if there are enough files in the
        # RawData directory, but that isn't implemented yet. For now you'll
        # just get and error sometime down the road when a period for the
        # missing files is needed.
        isRawDataDir = 'RawData' in os.listdir(self.directory)

        if isRawDataDir:
            print("Found the RawData directory:", rawDataDir)
            fname = bicycleName + 'Measured.txt'
            isMeasuredFile = fname in os.listdir(rawDataDir)
        else:
            isMeasuredFile = False

        isBenchmark = 'Benchmark' in self.parameters.keys()

        # the user wants to force a recalc and the data is there
        conOne = forceRawCalc and isRawDataDir and isMeasuredFile
        # the user doesn't want to force a recalc and there are no benchmark
        # parameters
        conTwo = not forceRawCalc and not isBenchmark

        if conOne or conTwo:
            print("Recalcuting the parameters.")
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
            print((stmt1 + stmt2) % self.bicycleName)
            # load the measured.txt file if it exists
            pathToRawFile = os.path.join(rawDataDir, self.bicycleName +
                                         'Measured.txt')
            try:
                self.parameters['Measured'] = \
                        io.load_parameter_text_file(pathToRawFile)
            except IOError:
                pass
        else:
            print('''There is no data available. Create
            bicycles/{sn}/Parameters/{sn}Benchmark.txt and/or fill
            bicycle/{sn}/RawData/ with pendulum data mat files and the
            {sn}Measured.txt file'''.format(sn=bicycleName))

    def __str__(self):
        if self.hasRider:
            desc = "{0} with {1} on board.".format(self.bicycleName,
                                                   self.riderName)
        else:
            desc = "{0} with no one on board.".format(self.bicycleName)
        return desc

    def save_parameters(self, filetype='text'):
        """Saves all the parameter sets to file.

        Parameters
        ==========

        filetype : string, optional
            - 'text' : a text file with parameters as ``c = 0.10+/-0.01``
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
        try:
            if os.path.isdir(photoDir):
                os.system('eog ' + os.path.join(photoDir, '*.*'))
            else:
                print("There are no photos of your bicycle.")
        except:
            raise NotImplementedError("This works only works for linux with " +
                                      "Eye of Gnome installed.")

    def steer_assembly_moment_of_inertia(self, handlebar=True, fork=True,
                                         wheel=True, aboutSteerAxis=False,
                                         nominal=False):
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

        This function does not currently take into account the flywheel, D, if
        it is defined, beware.

        """
        # load in the Benchmark parameter set
        par = self.parameters['Benchmark']

        if 'mD' in par.keys():
            print("You have a flywheel defined. Beware that it is ignored in "
                  "the calculations and the results do not reflect that it is "
                  "there.")

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

            # this is the inertia of the assembly about a reference frame
            # aligned with the steer axis and through the center of mass
            iAssRot = inertia.rotate_inertia_tensor(iAss, par['lam'])

        else:  # don't add the wheel
            mAss = m
            cAss = np.array([x, 0., z])
            iAssRot = inertia.rotate_inertia_tensor(I, par['lam'])

        if aboutSteerAxis:
            # this is the distance from the assembly com to the steer axis
            distance = geometry.distance_to_steer_axis(par['w'], par['c'],
                                                       par['lam'], cAss)
            print("handlebar cg distance", distance)

            # now calculate the inertia about the steer axis of the rotated
            # frame
            iAss = inertia.parallel_axis(iAssRot, mAss,
                                         np.array([distance, 0., 0.]))
        else:
            iAss = iAssRot

        if nominal:
            return unumpy.nominal_values(iAss)
        else:
            return iAss

    def calculate_from_measured(self, forcePeriodCalc=False):
        '''Calculates the parameters from measured data.'''

        rawDataDir = os.path.join(self.directory, 'RawData')
        pathToRawFile = os.path.join(rawDataDir,
                                     self.bicycleName + 'Measured.txt')

        # load the measured parameters
        self.parameters['Measured'] = io.load_parameter_text_file(
            pathToRawFile)

        forkIsSplit = is_fork_split(self.parameters['Measured'])

        # if the the user doesn't specifiy to force period calculation, then
        # see if enough data is actually available in the *Measured.txt file to
        # do the calculations
        if not forcePeriodCalc:
            forcePeriodCalc = period.check_for_period(
                self.parameters['Measured'], forkIsSplit)

        if forcePeriodCalc:
            # get the list of mat files associated with this bike
            matFiles = [x for x in os.listdir(rawDataDir)
                        if x.endswith('.mat')]
            matFiles.sort()
            # calculate the period for each file for this bicycle
            periods = period.calc_periods_for_files(rawDataDir, matFiles,
                                                    forkIsSplit)
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
            reCalc = True

        # first check to see if a rider has already been added
        if self.hasRider:
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

            if reCalc:
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
                                             riderName + self.bicycleName +
                                             'Benchmark.txt')
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
                        print('The measured bicycle parameters need to be '
                              'available, create your bicycle such that they '
                              'are available.')
                        raise
                    riderPar, human, bicycleRiderPar =\
                        rider.configure_rider(pathToRider, bicycleName,
                                              bicyclePar, measuredPar, draw)
                else:
                    print("Loaded the precalculated parameters from " +
                          "{0}".format(pathToParFile))
                    bicycleRiderPar = inertia.combine_bike_rider(bicyclePar,
                                                                 riderPar)
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
        """Returns a figure showing the basic bicycle geometry, the centers of
        mass and the moments of inertia.

        Parameters
        ==========
        show : boolean, optional
            If true ``matplotlib.pyplot.show()`` will be called before exiting
            the function.
        pendulum : boolean, optional
            If true the axes of the torsional pendulum will be displayed (only
            useful if raw measurement data is availabe).
        centerOfMass : boolean, optional
            If true the mass center of each rigid body will be displayed.
        inertiaEllipse : boolean optional
            If true inertia ellipses for each rigid body will be displayed.

        Returns
        =======
        fig : matplotlib.pyplot.Figure

        Notes
        =====
        If the flywheel is defined, it's center of mass corresponds to the
        front wheel and is not depicted in the plot.

        """
        par = io.remove_uncertainties(self.parameters['Benchmark'])
        parts = get_parts_in_parameters(par)

        try:
            slopes = io.remove_uncertainties(self.extras['slopes'])
            intercepts = io.remove_uncertainties(self.extras['intercepts'])
            penInertias = io.remove_uncertainties(
                self.extras['pendulumInertias'])
        except AttributeError:
            pendulum = False

        fig, ax = plt.subplots()

        fig.set_size_inches([4.0*GOLDEN_RATIO, 4.0])

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
                elif part in 'FD':
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
                Imin = Ip2D[0]
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
        ax.plot(x, np.zeros_like(x), 'k')

        # plot the rear wheel
        c = plt.Circle((0., par['rR']), radius=par['rR'], fill=False)
        ax.add_patch(c)

        # plot the front wheel
        c = plt.Circle((par['w'], par['rF']), radius=par['rF'], fill=False)
        ax.add_patch(c)

        # plot the fundamental bike
        deex, deez = geometry.fundamental_geometry_plot_data(par)
        ax.plot(deex, -deez, 'k')

        # plot the steer axis
        dx3 = deex[2] + deez[2] * (deex[2] - deex[1]) / (deez[1] - deez[2])
        ax.plot([deex[2], dx3],  [-deez[2], 0.], 'k--')

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
                                                                    (xcom,
                                                                     zcom))
                    comLineLength = penInertias[part][i]
                    xPlus = comLineLength / 2. * np.cos(np.arctan(m))
                    x = np.array([xPoint - xPlus,
                                  xPoint + xPlus])
                    z = -m * x - b
                    ax.plot(x, z, color=partColors[part])
                    # label the pendulum lines with a number
                    ax.text(x[0], z[0], str(i + 1))

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
            ax.text(par['w'] + sRad, par['rF'] + sRad, 'F')
            # rear wheel CoM
            ax = com_symbol(ax, (0., par['rR']), sRad,
                            color=partColors['R'])
            ax.text(0. + sRad, par['rR'] + sRad, 'R')
            for j, part in enumerate([x for x in parts
                                      if x not in 'RFD']):
                xcom = par['x' + part]
                zcom = par['z' + part]
                ax = com_symbol(ax, (xcom, -zcom), sRad,
                                color=partColors[part])
                ax.text(xcom + sRad, -zcom + sRad, part)
            if 'H' not in parts:
                ax = com_symbol(ax, (par['xH'], -par['zH']), sRad)
                ax.text(par['xH'] + sRad, -par['zH'] + sRad, 'H')

        # if there is a rider on the bike, make a simple stick figure
        top_of_head = 0.0
        if self.human:
            human = self.human
            mpar = self.parameters['Measured']
            bpar = self.parameters['Benchmark']
            # K2: lower leg, tip of foot to knee
            start = rider.yeadon_vec_to_bicycle_vec(human.K2.end_pos, mpar,
                                                    bpar)
            end = rider.yeadon_vec_to_bicycle_vec(human.K2.pos, mpar, bpar)
            ax.plot([start[0, 0], end[0, 0]],
                    [-start[2, 0], -end[2, 0]], 'k')
            # K1: upper leg, knee to hip
            start = rider.yeadon_vec_to_bicycle_vec(human.K2.pos, mpar, bpar)
            end = rider.yeadon_vec_to_bicycle_vec(human.K1.pos, mpar, bpar)
            ax.plot([start[0, 0], end[0, 0]],
                    [-start[2, 0], -end[2, 0]], 'k')
            # torso
            start = rider.yeadon_vec_to_bicycle_vec(human.K1.pos, mpar, bpar)
            end = rider.yeadon_vec_to_bicycle_vec(human.B1.pos, mpar, bpar)
            ax.plot([start[0, 0], end[0, 0]],
                    [-start[2, 0], -end[2, 0]], 'k')
            # B1: upper arm
            start = rider.yeadon_vec_to_bicycle_vec(human.B1.pos, mpar, bpar)
            end = rider.yeadon_vec_to_bicycle_vec(human.B2.pos, mpar, bpar)
            ax.plot([start[0, 0], end[0, 0]],
                    [-start[2, 0], -end[2, 0]], 'k')
            # B2: lower arm, elbow to tip of fingers
            start = rider.yeadon_vec_to_bicycle_vec(human.B2.pos, mpar, bpar)
            end = rider.yeadon_vec_to_bicycle_vec(human.B2.end_pos, mpar, bpar)
            ax.plot([start[0, 0], end[0, 0]],
                    [-start[2, 0], -end[2, 0]], 'k')
            # C: chest/head
            start = rider.yeadon_vec_to_bicycle_vec(human.B1.pos, mpar, bpar)
            end = rider.yeadon_vec_to_bicycle_vec(human.C.end_pos, mpar, bpar)
            ax.plot([start[0, 0], end[0, 0]],
                    [-start[2, 0], -end[2, 0]], 'k')
            top_of_head = -end[2, 0]

        ax.set_aspect('equal')

        # set the y limits to encompass the bicycle and rider geometry
        max_y = max([2*par['rR'],  # rear wheel diameter
                     2*par['rF'],  # front wheel diameter
                     max(-deez),  # max of Z values of bicycle geometry
                     top_of_head])  # max of Z values of human head
        min_y = min(-deez)
        if min_y >= 0.0:
            y_low = min([0.0, min_y])
        else:
            y_low = -np.ceil(np.abs(min_y))
        ax.set_ylim((y_low, np.ceil(max_y)))

        ax.set_title("{}\nBicycle Geometry".format(self.bicycleName))

        ax.set_xlabel('x [m]')
        ax.set_ylabel('-z [m]')

        if show:
            fig.show()

        # TODO : This should return ax instead of fig to follow typical
        # practice in other Python libraries.

        return fig

    def _plot_bicycle_geometry_plotly(self, show=True, pendulum=True,
                                      centerOfMass=True, inertiaEllipse=True):
        """Returns a Plotly figure showing the basic bicycle geometry,
        the centers of
        mass and the moments of inertia.

        Parameters
        ==========
        show : optional
            If true plotly figure will show.
        centerOfMass : boolean, optional
            If true the mass center of each rigid body will be displayed. but
            will have no trace in the legend since it already has a button.
            The hoverfunction will show where the COM's are located.
        inertiaEllipse : boolean optional
            If true inertia ellipses for each rigid body will be displayed.
            In some cases the ellipses are so large that they will not fit in
            the figure. Therefor the axis of this plot are fixed.

        Returns
        =======
        fig1 : A plotly figure

        """
        if px is None:
            raise ImportError('plotly is not installed')
        par = io.remove_uncertainties(self.parameters['Benchmark'])
        parts = get_parts_in_parameters(par)

        try:
            slopes = io.remove_uncertainties(self.extras['slopes'])
            intercepts = io.remove_uncertainties(self.extras['intercepts'])
            penInertias = io.remove_uncertainties(
                self.extras['pendulumInertias'])
        except AttributeError:
            pendulum = False

        fig1 = go.Figure()

        # define some colors for the parts
        # cmap = px.colors.sequential.Agsunset
        cmap = px.colors.qualitative.Pastel
        partColors = {}

        for i, part in enumerate(parts):
            partColors[part] = cmap[i]

        if inertiaEllipse:
            # plot the principal moments of inertia
            for j, part in enumerate(parts):
                I = inertia.part_inertia_tensor(par, part)
                Ip, C = inertia.principal_axes(I)
                if part == 'R':
                    center = np.array([0., par['rR']])
                elif part in 'FD':
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
                Imin = Ip2D[0]
                Imax = Ip2D[1]
                # get width and height of a ellipse with
                # the major axis equal to one
                unitWidth = 1. / 2. / np.sqrt(Imin) * np.sqrt(Imin)
                unitHeight = 1. / 2. / np.sqrt(Imax) * np.sqrt(Imin)
                # now scaled the width and height relative to
                # the maximum principal moment of inertia
                width = Imax * unitWidth
                height = Imax * unitHeight
                angle = -np.degrees(np.arccos(C2D[0, 0]))
                x_center = center[0]
                y_center = center[1]
                x_ep, y_ep = plot._generate_ellipse_plot_data(
                    x_center=x_center, y_center=y_center,
                    ax1=[np.cos(angle), np.sin(angle)],
                    ax2=[-np.sin(angle), np.cos(angle)],
                    a=height, b=width, N=100)

                fig1.add_scatter(x=x_ep, y=y_ep, mode='lines',
                                 name='Inertia of ' + part,
                                 line_color=partColors[part],
                                 fill='toself', opacity=0.5)

        # plot the ground line
        x = np.array([-par['rR'], par['w'] + par['rF']])
        fig1.add_trace(go.Scatter(x=x, y=np.zeros_like(x),
                       mode='lines',
                       name='Ground',
                       line_color='lightgrey',
                       hovertemplate="%{x:.3f}<br>%{y:.3f}"))

        def make_circle_legend(R, x_center_wheel, y_center_wheel):
            t = np.linspace(0, 2*np.pi, 100)
            xwh = R*np.cos(t)
            ywh = R*np.sin(t)
            x_wheel = xwh + x_center_wheel
            y_wheel = ywh + y_center_wheel
            return x_wheel, y_wheel

        # plot the rear wheel
        x_wheel_R, y_wheel_R = make_circle_legend(par['rR'], 0, par['rR'])
        fig1.add_trace(go.Scatter(x=x_wheel_R,
                                  y=y_wheel_R,
                                  mode='lines',
                                  line_color='grey',
                                  name='Rear wheel',
                                  hovertemplate="%{x:.3f}<br>%{y:.3f}"))

        # plot the front wheel
        x_wheel_F, y_wheel_F = make_circle_legend(par['rF'], par['w'],
                                                  par['rF'])
        fig1.add_trace(go.Scatter(x=x_wheel_F,
                                  y=y_wheel_F,
                                  mode='lines',
                                  line_color='grey',
                                  hovertemplate="%{x:.3f}<br>%{y:.3f}",
                                  name='Front wheel'))

        # plot the fundamental bike
        deex, deez = geometry.fundamental_geometry_plot_data(par)
        fig1.add_trace(go.Scatter(x=deex, y=-deez,
                                  mode='lines',
                                  name='Bicycle',
                                  line_color='black',
                                  hovertemplate="%{x:.3f}<br>%{y:.3f}"))

        # plot the steer axis
        dx3 = deex[2] + deez[2] * (deex[2] - deex[1]) / (deez[1] - deez[2])

        fig1.add_trace(go.Scatter(x=[deex[2], dx3], y=[-deez[2], 0.],
                                  mode='lines',
                                  name='Steer axis',
                                  hovertemplate="%{x:.3f}<br>%{y:.3f}",
                                  line=dict(dash='dash', color='dodgerblue')))

        # Update Layout so circle will be round and background white and no
        # grid
        fig1.update_xaxes(showgrid=False, zeroline=False)
        fig1.update_yaxes(showgrid=False, zeroline=False)

        fig1.update_xaxes(
            range=[-par['rR'], par['w']+par['rF']],  # sets the range of xaxis
            constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
        )
        fig1.update_yaxes(scaleanchor="x", scaleratio=1)
        fig1.update_layout(plot_bgcolor="white")

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
                                                                    (xcom,
                                                                     zcom))
                    comLineLength = penInertias[part][i]
                    xPlus = comLineLength / 2. * np.cos(np.arctan(m))
                    xp = np.array([xPoint - xPlus,
                                  xPoint + xPlus])
                    zp = -m*xp - b

                    fig1.add_scatter(x=xp, y=zp, mode='lines', name='Pendulum',
                                     line=dict(dash='dash'))

        if centerOfMass:
            def com_symbol(R, x_center, y_center, partcolor):
                t = np.linspace(0, 0.5*np.pi, 100)
                xs = R*np.cos(t)
                ys = R*np.sin(t)
                xc1 = xs + x_center
                yc1 = ys + y_center
                t2 = np.linspace(0.5*np.pi, np.pi, 100)
                xs2 = R*np.cos(t2)
                ys2 = R*np.sin(t2)
                xc2 = xs2 + x_center
                yc2 = ys2 + y_center
                t3 = np.linspace(np.pi, 1.5*np.pi, 100)
                xs3 = R*np.cos(t3)
                ys3 = R*np.sin(t3)
                xc3 = xs3 + x_center
                yc3 = ys3 + y_center
                t4 = np.linspace(1.5*np.pi, 2*np.pi, 100)
                xs4 = R*np.cos(t4)
                ys4 = R*np.sin(t4)
                xc4 = xs4 + x_center
                yc4 = ys4 + y_center

                fig1.add_trace(go.Scatter(x=[x_center, x_center+R],
                                          y=[y_center, y_center], mode='lines',
                                          line_color=partcolor,
                                          showlegend=False, hoverinfo='none'))
                fig1.add_trace(go.Scatter(x=[x_center, x_center],
                                          y=[y_center, y_center + R],
                                          mode='lines', line_color=partcolor,
                                          showlegend=False,
                                          hoverinfo='none'))
                fig1.add_trace(go.Scatter(x=xc1, y=yc1, mode='lines',
                                          line_color=partcolor,
                                          showlegend=False, fill='tonexty',
                                          hoverinfo='none'))
                fig1.add_trace(go.Scatter(x=xc2, y=yc2, mode='lines',
                                          line_color=partcolor,
                                          showlegend=False, hoverinfo='none'))
                fig1.add_trace(go.Scatter(x=xc3, y=yc3, mode='lines',
                                          line_color=partcolor,
                                          showlegend=False, hoverinfo='none'))
                fig1.add_trace(go.Scatter(x=[x_center-R, x_center],
                                          y=[y_center, y_center], mode='lines',
                                          line_color=partcolor,
                                          showlegend=False, fill='tonexty',
                                          hoverinfo='none'))
                fig1.add_trace(go.Scatter(x=[x_center, x_center],
                                          y=[y_center-R, y_center],
                                          mode='lines', line_color=partcolor,
                                          showlegend=False,
                                          hoverinfo='none'))
                fig1.add_trace(go.Scatter(x=xc4, y=yc4, mode='lines',
                                          line_color=partcolor,
                                          showlegend=False, hoverinfo='none'))
                fig1.add_trace(go.Scatter(x=[x_center, x_center],
                                          y=[y_center, y_center],
                                          mode='lines', line_color=partcolor,
                                          hovertemplate="%{x:.3f}<br>%{y:.3f}",
                                          name='COM', showlegend=False))
                return fig1

            # radius of the CoM symbol
            sRad = 0.03
            # front wheel CoM
            x_com_Wf = par['w']
            y_com_Wf = par['rF']
            fig1 = com_symbol(sRad, x_com_Wf, y_com_Wf, partColors['F'])
            fig1.add_annotation(text='F', xref='x', yref='y',
                                x=x_com_Wf + 0.055,
                                y=y_com_Wf+0.055, showarrow=False,
                                font=dict(size=15))

            # rear wheel CoM
            fig1 = com_symbol(sRad, 0., par['rR'], partColors['R'])
            fig1.add_annotation(text='R', xref='x', yref='y', x=0.055,
                                y=par['rR']+0.055, showarrow=False,
                                font=dict(size=15))

            for j, part in enumerate([x for x in parts
                                      if x not in 'RFD']):
                xcom = par['x' + part]
                zcom = par['z' + part]
                fig1 = com_symbol(sRad, xcom, -zcom, partColors[part])
                fig1.add_annotation(text=part, xref='x', yref='y',
                                    x=xcom+0.055, y=-zcom+0.055,
                                    showarrow=False, font=dict(size=15))

            if 'H' not in parts:
                fig1 = com_symbol(sRad, par['xH'], -par['zH'], partColors['H'])
                fig1.add_annotation(text="H", xref='x', yref='y',
                                    x=par['xH']+0.055, y=-par['zH']+0.055,
                                    showarrow=False, font=dict(size=15))

        # if there is a rider on the bike, make a simple stick figure
        if self.human:
            human = self.human
            mpar = self.parameters['Measured']
            bpar = self.parameters['Benchmark']
            # K2: lower leg, tip of foot to knee
            start = rider.yeadon_vec_to_bicycle_vec(human.K2.end_pos, mpar,
                                                    bpar)
            end = rider.yeadon_vec_to_bicycle_vec(human.K2.pos, mpar, bpar)
            fig1.add_trace(go.Scatter(x=[start[0, 0], end[0, 0]],
                                      y=[-start[2, 0], -end[2, 0]],
                                      mode='lines'))

            # K1: upper leg, knee to hip
            start = rider.yeadon_vec_to_bicycle_vec(human.K2.pos, mpar, bpar)
            end = rider.yeadon_vec_to_bicycle_vec(human.K1.pos, mpar, bpar)

        fig1.update_layout(title=dict(text='Bicycle geometry',
                           font=dict(family="Segoe UI", size=25)),
                           yaxis_title='z-axis [m]',
                           font_family="Source Sans Pro",
                           hoverlabel=dict(font_family="Source Sans Pro"))

        fig1.update_layout(yaxis=dict(autorange=True, showgrid=False,
                                      ticks='outside', showticklabels=True))
        fig1.update_layout(xaxis=dict(autorange=True, showgrid=False, ticks='',
                                      showticklabels=False))
        if show:
            fig1.show()
        return fig1

    def canonical(self, nominal=False):
        """
        Returns the canonical velocity and gravity independent matrices for
        the Whipple bicycle model linearized about the nominal
        configuration.

        Parameters
        ----------
        nominal : boolean, optional
            The default is false and uarrays are returned with the
            calculated uncertainties. If true ndarrays are returned without
            uncertainties.

        Returns
        -------
        M : uarray, shape(2,2)
            Mass matrix.
        C1 : uarray, shape(2,2)
            Velocity independent damping matrix.
        K0 : uarray, shape(2,2)
            Gravity independent part of the stiffness matrix.
        K2 : uarray, shape(2,2)
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

        If you have a flywheel defined, body D, it will completely be
        ignored in these results. These results are strictly for the Whipple
        bicycle model.

        """

        par = self.parameters['Benchmark']

        M, C1, K0, K2 = bicycle.benchmark_par_to_canonical(par)

        if nominal is True:
            return (unumpy.nominal_values(M),
                    unumpy.nominal_values(C1),
                    unumpy.nominal_values(K0),
                    unumpy.nominal_values(K2))
        elif nominal is False:
            return M, C1, K0, K2
        else:
            raise ValueError('nominal must be True or False')

    def state_space(self, speed, nominal=False):
        """
        Returns the A and B matrices for the Whipple model linearized about
        the upright constant velocity configuration.


        Parameters
        ----------
        speed : float
            The speed of the bicycle.
        nominal : boolean, optional
            The default is false and uarrays are returned with the calculated
            uncertainties. If true ndarrays are returned without uncertainties.

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

        The states are [roll angle,
                        steer angle,
                        roll rate,
                        steer rate]

        The inputs are [roll torque,
                        steer torque]

        If you have a flywheel defined, body D, it will completely be ignored
        in these results. These results are strictly for the Whipple bicycle
        model.

        """

        M, C1, K0, K2 = self.canonical()

        g = self.parameters['Benchmark']['g']

        A, B = bicycle.ab_matrix(M, C1, K0, K2, speed, g)

        if nominal is True:
            return (unumpy.nominal_values(A), unumpy.nominal_values(B))
        elif nominal is False:
            return A, B
        else:
            raise ValueError('nominal must be True or False')

    def eig(self, speeds):
        '''Returns the eigenvalues and eigenvectors of the Whipple bicycle
        model linearized about the nominal configuration.

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

        Notes
        -----
        If you have a flywheel defined, body D, it will completely be ignored
        in these results. These results are strictly for the Whipple bicycle
        model.

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
                                  color='black', show=False, largest=False,
                                  linestyle='-', grid=False, show_legend=True):
        """Returns a plot of the eigenvalues versus speed for the current
        benchmark parameters.

        Parameters
        ----------
        speeds : ndarray, shape(n,)
            An array of speeds to calculate the eigenvalues at.
        fig : matplotlib figure, optional
            A figure to plot to.
        generic : boolean
            If true the lines will all be the same color and the modes will
            not be labeled.
        color : matplotlib color
            If generic is true this will be the color of the plot lines.
        largest : boolean
            If true, only the largest eigenvalue is plotted.
        grid : boolean, optional
            If true, displays a grid on the plot.
        show_legend: boolean, optional
            If true, displays a legend describing the different parts of the
            solution shown.

        Returns
        -------
        fig : matpolib.pyplot.Figure
            The figure.

        Notes
        -----
        If you have a flywheel defined, body D, it will completely be
        ignored in these results. These results are strictly for the Whipple
        bicycle model.

        """

        # sort the speeds in case they aren't
        speeds = np.sort(speeds)

        # figure properties
        fig_height = 4.0  # inches
        figsize = [fig_height*GOLDEN_RATIO, fig_height]
        params = {
            'axes.labelsize': 8,
            'text.fontsize': 10,
            'legend.fontsize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'figure.figsize': figsize
            }
        # NOTE : text.fontsize no longer supported in matplotlib
        try:
            plt.rcParams.update(params)
        except KeyError:
            del params['text.fontsize']
            params['font.size'] = 10
            plt.rcParams.update(params)

        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)

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
            ax.plot(speeds, maxEval, color=color, label=maxLabel,
                    linestyle=linestyle, linewidth=1.5)
            # x axis line
            ax.plot(speeds, np.zeros_like(speeds), 'k-', label='_nolegend_',
                    linewidth=1.5)
            ax.set_ylim((np.min(maxEval), np.max(maxEval)))
            ax.set_ylabel('Real Part of the Largest Eigenvalue [1/s]')
        else:
            wea, cap, cas = bicycle.sort_modes(evals, evecs)

            # imaginary components
            ax.plot(speeds, np.abs(np.imag(wea['evals'])), color=weaveColor,
                    label=legend[0], linestyle='--')
            ax.plot(speeds, np.abs(np.imag(cap['evals'])), color=capsizeColor,
                    label=legend[1], linestyle='--')
            ax.plot(speeds, np.abs(np.imag(cas['evals'])), color=casterColor,
                    label=legend[2], linestyle='--')

            # x axis line
            ax.plot(speeds, np.zeros_like(speeds), 'k-', label='_nolegend_',
                    linewidth=1.5)

            # plot the real parts of the eigenvalues
            ax.plot(speeds, np.real(wea['evals']), color=weaveColor,
                    label=legend[3])
            ax.plot(speeds, np.real(cap['evals']), color=capsizeColor,
                    label=legend[4])
            ax.plot(speeds, np.real(cas['evals']), color=casterColor,
                    label=legend[5])

            # set labels and limits
            ax.set_ylabel('Real and Imaginary Parts of the Eigenvalue [1/s]')

        ax.set_xlim((speeds[0], speeds[-1]))
        ax.set_xlabel('Speed [m/s]')

        if generic:
            ax.set_title('Eigenvalues vs Speed')
        else:
            ax.set_title('%s\nEigenvalues vs Speed' % self.bicycleName)
            if show_legend:
                ax.legend()

        if grid:
            ax.grid()

        if show:
            fig.show()

        return fig

    def _plot_eigenvalues_vs_speed_plotly(self, speeds, fig=None, show=True,
                                          largest=False,
                                          stability_region=True):
        if px is None:
            raise ImportError('plotly is not installed')
        speeds = np.sort(speeds)
        if fig is None:
            fig = go.Figure(layout_yaxis_range=[-10, 10])
            evals, evecs = self.eig(speeds)

        if largest:
            fig.add_trace(go.Scatter(x=speeds, y=np.max(evals)))
            fig.show()
        else:
            w, cap, cas = bicycle.sort_modes(evals, evecs)
            colors_eig = px.colors.qualitative.Pastel
            weaveColor1 = colors_eig[0]
            weaveColor2 = colors_eig[1]
            capsizeColor = colors_eig[2]
            casterColor = colors_eig[5]
        wea1 = w['evals'][:, 0]
        wea2 = w['evals'][:, 1]
        fig.add_trace(go.Scatter(x=speeds, y=np.real(wea1),
                                 mode='lines',
                                 name='Re Weave',
                                 line=dict(color=weaveColor1),
                                 text='Re'))
        fig.add_trace(go.Scatter(x=speeds, y=np.real(wea2),
                                 mode='lines',
                                 name='Re Weave',
                                 line=dict(color=weaveColor2),
                                 text='Re'))
        fig.add_trace(go.Scatter(x=speeds, y=np.real(cap['evals']),
                                 mode='lines',
                                 name='Re Capsize',
                                 line=dict(color=capsizeColor),
                                 text='Re'))
        fig.add_trace(go.Scatter(x=speeds, y=np.real(cas['evals']),
                                 mode='lines',
                                 name='Re Castering',
                                 line=dict(color=casterColor),
                                 text='Re'))
        fig.add_trace(go.Scatter(x=speeds, y=np.abs(np.imag(wea1)),
                                 mode='lines',
                                 name='Im Weave',
                                 line=dict(color=weaveColor1, dash='dash'),
                                 text='Im'))
        fig.add_trace(go.Scatter(x=speeds, y=np.abs(np.imag(wea2)),
                                 mode='lines',
                                 name='Im Weave',
                                 line=dict(color=weaveColor2, dash='dash'),
                                 text='Im'))
        fig.add_trace(go.Scatter(x=speeds, y=np.abs(np.imag(cap['evals'])),
                                 mode='lines',
                                 name='Im Capsize',
                                 line=dict(color=capsizeColor, dash='dash'),
                                 text='Im'))
        fig.add_trace(go.Scatter(x=speeds, y=np.abs(np.imag(cas['evals'])),
                                 mode='lines',
                                 name='Im Castering',
                                 line=dict(color=casterColor, dash='dash'),
                                 text='Im'))
        if stability_region:
            try:
                v_start_stab = max([min(speeds[np.real(wea2) < 0]),
                                    min(speeds[np.real(cas['evals']) < 0]),
                                    min(speeds[np.real(cap['evals']) < 0]),
                                    min(speeds[np.real(wea1) < 0],
                                        default="EMPTY")])
                v_end_stab = min([max(speeds[np.real(wea2) < 0]),
                                  max(speeds[np.real(cas['evals']) < 0]),
                                  max(speeds[np.real(cap['evals']) < 0]),
                                  max(speeds[np.real(wea1) < 0])])
            except:  # TODO : Add explicit exception
                fig.add_annotation(x=0.5*max(speeds), y=9,
                                   text="No stability region",
                                   showarrow=False)
            if (v_start_stab > v_end_stab):
                fig.add_annotation(x=0.5*max(speeds), y=9,
                                   text="No stability region",
                                   showarrow=False)
            elif (v_end_stab - v_start_stab < 0.0001):
                fig.add_annotation(x=0.5*max(speeds), y=9,
                                   text="No stability region",
                                   showarrow=False)
            else:
                fig.add_vrect(x0=v_start_stab, x1=v_end_stab,
                              annotation_text="Self stability",
                              annotation_position='top left',
                              # fillcolor="blue", opacity=0.25,
                              fillcolor='rgba(71,147,231,0.5)',
                              line_width=0, row=1, col=1)

        fig.update_layout(title=dict(text='Eigenvalues vs velocity',
                          font=dict(family="Segoe UI", size=25)),
                          font_family="Source Sans Pro",
                          plot_bgcolor='rgba(39,128,227,0.15)',
                          xaxis_title='Velocity [m/s]',
                          yaxis_title='Eigenvalues [1/s]',
                          hoverlabel=dict(font_family="Source Sans Pro"))
        fig.update_traces(hovertemplate="%{x:.3f}<br>%{y:.3f}")

        if show:
            fig.show()

        return fig

    def plot_bode(self, speed, u, y, **kwargs):
        """Returns a Bode plot.

        Parameters
        ----------
        speed : float
            The speed at which to evaluate the system.
        u : integer
            An integer between 0 and 1 corresponding to the inputs roll torque
            and steer torque.
        y : integer
            An integer between 0 and 3 corresponding to the inputs roll angle
            steer angle, roll rate, steer rate.
        kwargs : keyword pairs
            Any options that can be passed to dtk.bode.

        Returns
        -------
        mag : ndarray, shape(1000,)
            The magnitude in dB of the frequency reponse.
        phase : ndarray, shape(1000,)
            The phase in degress of the frequency response.
        fig : matplotlib figure
            The Bode plot.

        """

        A, B = self.state_space(speed, nominal=True)

        C = np.eye(A.shape[0])
        D = np.zeros_like(B)

        w = np.logspace(0, 2, 1000)

        outputNames = ['Roll Angle', 'Steer Angle', 'Roll Rate', 'Steer Rate']
        inputNames = ['Roll Torque', 'Steer Torque']

        if 'title' not in kwargs.keys():
            kwargs['title'] = inputNames[u] + ' to ' + outputNames[y]

        bode = control.bode((A, B[:, u], C[y, :], D[y, u]), w, **kwargs)

        return bode

    def compare_bode_speeds(self, speeds, u, y, fig=None):
        """Returns a figure with the Bode plots of multiple bicycles.

        Parameters
        ----------
        speeds : list
            A list of speeds at which to evaluate the system.
        u : integer
            An integer between 0 and 1 corresponding to the inputs roll torque
            and steer torque.
        y : integer
            An integer between 0 and 3 corresponding to the inputs roll angle,
            steer angle, roll rate, steer rate.

        Returns
        -------
        fig : matplotlib.Figure instance
            The Bode plot.

        Notes
        -----
        The phases are matched around zero degrees at with respect to the first
        frequency.

        """

        if fig is None:
            fig = plt.figure()

        for speed in speeds:
            self.plot_bode(speed, u, y, label=str(speed) + ' m/s', fig=fig)

        # take care of phase misalignment
        phaseLines = fig.ax2.lines
        for line in phaseLines:
            firstValue = line.get_ydata()[0]
            n = np.ceil(np.floor(abs(firstValue / 180.)) / 2.)
            line.set_ydata(line.get_ydata() - np.sign(firstValue) * n * 360.)
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
        'S', 'G', 'D'.

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
    try:
        # we measured the mass of the flywheel plus the mass of the front
        # wheel, mp['mD'], so to get the actual mass of the flywheel, subtract
        # the mass of the front wheel
        par['mD'] = mp['mD'] - mp['mF']
    except KeyError:
        pass
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
    try:
        # we measured the inertia of the front wheel with the flywheel inside
        iFlywheelPlusFwheel = inertia.compound_pendulum_inertia(
            mp['mD'], mp['g'], mp['lF'], mp['TcD1'])
        par['IDyy'] = iFlywheelPlusFwheel - par['IFyy']
    except KeyError:
        pass

    # calculate the y inertias for the frame and fork
    lB = (par['xB']**2 + (par['zB'] + par['rR'])**2)**(0.5)
    par['IByy'] = inertia.compound_pendulum_inertia(mp['mB'], mp['g'], lB,
                                                    mp['TcB1'])

    if forkIsSplit:
        # fork
        lS = ((par['xS'] - par['w'])**2 +
              (par['zS'] + par['rF'])**2)**(0.5)
        par['ISyy'] = inertia.compound_pendulum_inertia(mp['mS'], mp['g'], lS,
                                                        mp['TcS1'])
        # handlebar
        l1, l2 = geometry.calculate_l1_l2(mp['h6'], mp['h7'], mp['d5'],
                                          mp['d6'], mp['l'])
        u1, u2 = geometry.fwheel_to_handlebar_ref(par['lam'], l1, l2)
        lG = ((par['xG'] - par['w'] + u1)**2 +
              (par['zG'] + par['rF'] + u2)**2)**(.5)
        par['IGyy'] = inertia.compound_pendulum_inertia(mp['mG'], mp['g'], lG,
                                                        mp['TcG1'])
    else:
        lH = ((par['xH'] - par['w'])**2 +
              (par['zH'] + par['rF'])**2)**(0.5)
        par['IHyy'] = inertia.compound_pendulum_inertia(mp['mH'], mp['g'], lH,
                                                        mp['TcH1'])

    # calculate the stiffness of the torsional pendulum
    IPxx, IPyy, IPzz = inertia.tube_inertia(mp['lP'], mp['mP'],
                                            mp['dP'] / 2., 0.)
    torStiff = inertia.torsional_pendulum_stiffness(IPyy, mp['TtP1'])

    # calculate the wheel x/z inertias
    par['IFxx'] = inertia.tor_inertia(torStiff, mp['TtF1'])
    par['IRxx'] = inertia.tor_inertia(torStiff, mp['TtR1'])
    try:
        par['IDxx'] = inertia.tor_inertia(torStiff, mp['TtD1']) - par['IFxx']
    except KeyError:
        pass

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
            penInertia[i] = inertia.tor_inertia(torStiff,
                                                mp['Tt' + part + str(i + 1)])
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
    extras = {'slopes': slopes,
              'intercepts': intercepts,
              'betas': betas,
              'pendulumInertias': pendulumInertias}

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
