"""
======================
Using Bicycles
======================

Loading bicycle data
====================

The easiest way to load a bicycle is:
"""

import numpy as np
import bicycleparameters as bp
from bicycleparameters import tables

bicycle = bp.Bicycle('Stratos', pathToData='../data')

# %%
# This will create an instance of the Bicycle class in the variable bicycle
# based ``./bicycles/Stratos/Parameters/``. If so, it loads the data, if not it
# looks calculate the periods. If no data is there, then you get an error.
# doesn't then the program will look for the series of ``.mat`` files need to
# for ``./bicycles/Stratos/RawData/StratosMeasurments.txt`` so that it can
# generate the parameter set. The raw measurement file may or may not contain
# the looks to see if there are any parameter sets in off of input data from
# the ``./bicycles/Stratos/`` directory. The program first oscillation period
# data for the bicycle moment of inertia calculations. If it
#
# There are other loading options::
#
#     bicycle = bp.Bicycle('Stratos', pathToData='..', forceRawCalc=True,
#                          forcePeriodCalc=True)
#
# The ``pathToData`` option allows you specify a directory other than the
# current directory as your data directory. The ``forceRawCalc`` forces the
# program to load ``./bicycles/Stratos/RawData/StratosMeasurments.txt`` and
# recalculate the parameters regardless if there are any parameter files
# available in ``./bicycles/Stratos/Parameters/``. The ``forcePeriodCalc``
# option forces the period calculation from the ``.mat`` files regardless if
# they already exist in the raw measurement file.
#
# Exploring bicycle parameter data
# ================================
#
# The bicycle has a name:
bicycle.bicycleName

# %%
# and a directory where its data is stored:
bicycle.directory

# %%
# The benchmark bicycle parameters are the fundamental parameter set that is
# used behind the scenes for calculations. To access them type:
bPar = bicycle.parameters['Benchmark']
bPar['xB']

# %%
# The program automatically calculates the uncertainties in the parameters
# based on the raw measurements or the uncertainties provided in the parameter
# files.  If you'd like to work with the pure values you can remove them:
bParPure = bp.io.remove_uncertainties(bPar)
bParPure['xB']

# %%
# That goes the same for all values with uncertainties. Check out the
# `uncertainties <http://packages.python.org/uncertainties>`_ package details
# for more ways to manipulate the quantities.
#
# If the bicycle was calculated from raw data measurements you can access them
# by:
bicycle.parameters['Measured']

# %%
# All parameter sets are stored in the parameter dictionary of the bicycle
# instance.
#
# To modify a parameter type:
bicycle.parameters['Benchmark']['mB'] = 50.

# %%
# You can regenerate the parameter sets from the raw data stored in the
# bicycle's directory by calling:
bicycle.calculate_from_measured()

# %%
# Basic Analysis
# ==============
# The program has some basic bicycle analysis tools based on the Whipple
# bicycle model which has been linearized about the upright configuration.
#
# The canonical matrices for the equations of motion can be computed:
M, C1, K0, K2 = bicycle.canonical()
M

# %%
C1

# %%
K0

# %%
K2

# %%
# as well as the state and input matrices for state space form at a particular
# speed (1.34 m/s):

A, B = bicycle.state_space(1.34)
A

# %%
B

# %%
# You can calculate the eigenvalues and eigenvectors at any speed by calling:
w, v = bicycle.eig(4.28)  # the speed should be in meters/second

# %%
# eigenvalues
w

# %%
# eigenvectors
v

# %%
# The ``eig`` function also accepts a one dimensional array of speeds and
# returns eigenvalues for all speeds. Note that uncertainty propagation into
# the eigenvalue calculations is not supported yet.
#
# The moment of inertia of the steer assembly (handlebar, fork and/or front
# wheel) can be computed either about the center of mass or a point on the
# steer axis, both with reference to a frame aligned with the steer axis:
bicycle.steer_assembly_moment_of_inertia(aboutSteerAxis=True)

# %%
# Plots
# -----
# You can plot the geometry of the bicycle and include the mass centers of the
# various bodies, the inertia ellipsoids and the torsional pendulum axes from
# the raw measurement data:
bicycle.plot_bicycle_geometry()

# %%
# For visualization of the linear analysis you can plot the root loci of the
# real and imaginary parts of the eigenvalues as a function of speed:
speeds = np.linspace(0., 10., num=100)
bicycle.plot_eigenvalues_vs_speed(speeds)

# %%
# You can also compare the eigenvalues of two or more bicycles:
yellowrev = bp.Bicycle('Yellowrev', pathToData='../data')
bp.plot_eigenvalues([bicycle, yellowrev], speeds)

# %%
# Tables
# ------
# You can generate reStructuredText tables of the bicycle parameters with the
# ``Table`` class:
tab = tables.Table('Benchmark', False, (bicycle, yellowrev))
rst = tab.create_rst_table()
print(rst)

# %%
# Which renders in Sphinx like:
#
# +----------+------------------+------------------+
# |          | Stratos          | Yellowrev        |
# +==========+=========+========+=========+========+
# | Variable | v       | sigma  | v       | sigma  |
# +----------+---------+--------+---------+--------+
# | IBxx     | 0.373   | 0.002  | 0.2254  | 0.0009 |
# +----------+---------+--------+---------+--------+
# | IBxz     | -0.0383 | 0.0004 | 0.0179  | 0.0001 |
# +----------+---------+--------+---------+--------+
# | IByy     | 0.717   | 0.003  | 0.388   | 0.005  |
# +----------+---------+--------+---------+--------+
# | IBzz     | 0.455   | 0.002  | 0.2147  | 0.0009 |
# +----------+---------+--------+---------+--------+
# | IFxx     | 0.0916  | 0.0004 | 0.0852  | 0.0003 |
# +----------+---------+--------+---------+--------+
# | IFyy     | 0.157   | 0.001  | 0.147   | 0.002  |
# +----------+---------+--------+---------+--------+
# | IHxx     | 0.1768  | 0.0008 | 0.1475  | 0.0006 |
# +----------+---------+--------+---------+--------+
# | IHxz     | -0.0273 | 0.0006 | -0.0172 | 0.0005 |
# +----------+---------+--------+---------+--------+
# | IHyy     | 0.144   | 0.002  | 0.120   | 0.002  |
# +----------+---------+--------+---------+--------+
# | IHzz     | 0.0446  | 0.0003 | 0.0294  | 0.0004 |
# +----------+---------+--------+---------+--------+
# | IRxx     | 0.0939  | 0.0004 | 0.0877  | 0.0004 |
# +----------+---------+--------+---------+--------+
# | IRyy     | 0.154   | 0.001  | 0.149   | 0.001  |
# +----------+---------+--------+---------+--------+
# | c        | 0.056   | 0.002  | 0.180   | 0.002  |
# +----------+---------+--------+---------+--------+
# | g        | 9.81    | 0.01   | 9.81    | 0.01   |
# +----------+---------+--------+---------+--------+
# | lam      | 0.295   | 0.003  | 0.339   | 0.003  |
# +----------+---------+--------+---------+--------+
# | mB       | 7.22    | 0.02   | 3.31    | 0.02   |
# +----------+---------+--------+---------+--------+
# | mF       | 3.33    | 0.02   | 1.90    | 0.02   |
# +----------+---------+--------+---------+--------+
# | mH       | 3.04    | 0.02   | 2.45    | 0.02   |
# +----------+---------+--------+---------+--------+
# | mR       | 3.96    | 0.02   | 2.57    | 0.02   |
# +----------+---------+--------+---------+--------+
# | rF       | 0.3400  | 0.0001 | 0.3419  | 0.0001 |
# +----------+---------+--------+---------+--------+
# | rR       | 0.3385  | 0.0001 | 0.3414  | 0.0001 |
# +----------+---------+--------+---------+--------+
# | w        | 1.037   | 0.002  | 0.985   | 0.002  |
# +----------+---------+--------+---------+--------+
# | xB       | 0.326   | 0.003  | 0.412   | 0.004  |
# +----------+---------+--------+---------+--------+
# | xH       | 0.911   | 0.004  | 0.919   | 0.005  |
# +----------+---------+--------+---------+--------+
# | zB       | -0.483  | 0.003  | -0.618  | 0.004  |
# +----------+---------+--------+---------+--------+
# | zH       | -0.730  | 0.002  | -0.816  | 0.002  |
# +----------+---------+--------+---------+--------+
#
# Rigid Rider
# ===========
# The program also allows one to add the inertial affects of a rigid rider to
# the Whipple bicycle system.
#
# Rider Data
# ----------
# You can provide rider data in one of two ways, much in the same way as the
# bicycle. If you have the inertial parameters of a rider, e.g. Jason, simply
# add a file into the ``./riders/Jason/Parameters/`` directory. Or if you have
# raw measurements of the rider add the two files to
# ``./riders/Jason/RawData/``. The `yeadon documentation`_ explains how to
# collect the data for a rider.
#
# .. _yeadon documentation: http://packages.python.org/yeadon
#
# Adding a Rider
# --------------
# To add a rider key in:
bicycle.add_rider('Jason')

# %%
# The program first looks for a parameter for for Jason sitting on the Stratos
# and if it can't find one, it looks for the raw data for Jason and computes
# the inertial parameters. You can force calculation from raw data with:
bicycle = bp.Bicycle('Stratos', pathToData='../data')
bicycle.add_rider('Jason', reCalc=True)

# %%
# Exploring the rider
# -------------------
# The bicycle has a few new attributes now that it has a rider:
bicycle.hasRider

# %%
bicycle.riderName

# %%
bicycle.riderPar  # inertial parmeters of the rider

# %%
bicycle.human  # this is a yeadon.human object representing the Jason

# %%
# The bicycle parameters now reflect that a rigid rider has been added to the
# bicycle frame:
bicycle.parameters['Benchmark']['mB']

# %%
# At this point, the uncertainties don't necessarily offer much information for
# any of the parameters that are functions of the rider, because we do not have
# a good idea of the uncertainty in the human inertia calculations in the
# Yeadon method.
#
# Analysis
# --------
# The same linear analysis can be performed now that a rider has been added,
# albeit the reported values and graphs will reflect the fact that the bicycle
# frame has the added inertial effects of the rider.
#
# Plots
# -----
# The bicycle geometry plot now reflects that there is a rider on the bicycle
# and displays a simplified depiction:
bicycle.plot_bicycle_geometry()

# %%
# The eigenvalue plot also reflects the changes:
bicycle.plot_eigenvalues_vs_speed(speeds)

# %%
# Rider Visualization
# -------------------
# If you have the optional dependency, visual python, for yeadon installed then
# you can output a three dimensional picture of the Yeadon model configured to
# be seated on the bicycle. This is a bit buggy due to the nature of visual
# python, but is useful none-the-less.:
bicycle = bp.Bicycle('Stratos', pathToData='../data')
bicycle.add_rider('Jason', draw=True)
