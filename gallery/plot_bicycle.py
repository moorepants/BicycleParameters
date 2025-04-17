"""
==============
Using Bicycles
==============

"""
import pprint
from pathlib import Path

import numpy as np
import bicycleparameters as bp
from bicycleparameters import Bicycle, tables

# %%
# Loading Bicycle Data
# ====================
#
# To load the data from one of the bicycles in the data folder, instantiate a
# :py:class:`~bicycleparameters.main.Bicycle` object using the bicycle's name:
bicycle = Bicycle('Stratos', pathToData='../data')

# %%
# This will create an instance of the Bicycle class in the variable bicycle
# based off of input data from the ``./bicycles/Stratos/`` directory. The
# program first looks to see if there are any parameter sets in
# ``./bicycles/Stratos/Parameters/``. If so, it loads the data, if not it looks
# for ``./bicycles/Stratos/RawData/StratosMeasurments.txt`` so that it can
# generate the parameter set. The raw measurement file may or may not contain
# the oscillation period data for the bicycle moment of inertia calculations.
# If it doesn't then the program will look for the series of ``.mat`` files
# need to calculate the periods. If no data is there, then you get an error.
#
# There are other loading options::
#
#     bicycle = Bicycle('Stratos', pathToData='..', forceRawCalc=True,
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
# Exploring Bicycle Parameter Data
# ================================
#
# The bicycle has a name:
bicycle.bicycleName

# %%
# and a directory where its data is sourced:
bicycle.directory

# %%
# The benchmark bicycle parameters from [Meijaard2007]_ are the fundamental
# parameter set that is used behind the scenes for calculations. To access them
# type:
b_par = bicycle.parameters['Benchmark']
pprint.pprint(b_par)

# %%
b_par['xB']

# %%
# The program automatically calculates the uncertainties in the parameters
# based on the raw measurements or the uncertainties provided in the parameter
# files. If you'd like to work with the pure values you can remove them from
# the entire dictionary:
b_par_pure = bp.io.remove_uncertainties(b_par)
b_par_pure['xB']

# %%
# or any single uncertainity quantity's nominal value can be extracted with:
b_par['xB'].nominal_value

# %%
# That goes the same for all values with uncertainties. Check out the
# `uncertainties <http://packages.python.org/uncertainties>`_ package details
# for more ways to manipulate the quantities.
#
# If the bicycle was calculated from raw data measurements you can access them
# by:
pprint.pprint(bicycle.parameters['Measured'])

# %%
# All parameter sets are stored in the parameter dictionary of the bicycle
# instance, which is mutable. To modify a parameter type:
bicycle.parameters['Benchmark']['mB'] = 50.0
bicycle.parameters['Benchmark']['mB']

# %%
# You can regenerate the parameter sets from the raw data stored in the
# bicycle's directory by calling and resetting the parameters:
par, extra = bicycle.calculate_from_measured()
bicycle.parameters['Benchmark'] = par
bicycle.parameters['Benchmark']['mB']

# %%
# Basic Linear Model Analysis
# ===========================
#
# The :py:class:`~bicycleparameters.main.Bicycle` object has some basic bicycle
# analysis tools based on the linear Carvallo-Whipple bicycle model which has
# been linearized about the upright configuration. For example, the canonical
# coefficient matrices for the equations of motion can be computed:
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
# speed, here 1.34 m/s:
A, B = bicycle.state_space(1.34)
A

# %%
B

# %%
# You can calculate the eigenvalues and eigenvectors at any speed, e.g. 4.28
# m/s, by calling:
w, v = bicycle.eig(4.28)

# %%
# eigenvalues:
w

# %%
# eigenvectors:
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
# =====
# You can plot the geometry of the bicycle and include the mass centers of the
# various bodies, the inertia ellipsoids and the torsional pendulum axes from
# the raw measurement data:
_ = bicycle.plot_bicycle_geometry()

# %%
# A Bode plot for any input output pair can be generated with:
_ = bicycle.plot_bode(3.0, 1, 2)

# %%
# For visualization of the linear analysis you can plot the root loci of the
# real and imaginary parts of the eigenvalues as a function of speed:
speeds = np.linspace(0., 10., num=100)
_ = bicycle.plot_eigenvalues_vs_speed(speeds)

# %%
# You can also compare the eigenvalues of two or more bicycles:
yellowrev = Bicycle('Yellowrev', pathToData='../data')
_ = bp.plot_eigenvalues([bicycle, yellowrev], speeds)

# %%
# Tables
# ======
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
# Adding a Rigid Rider
# ====================
# The program also allows one to add the inertial affects of a rigid rider to
# the Whipple bicycle model.
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
# the inertial parameters. You can force calculation from raw data with::
#
#    bicycle.add_rider('Jason', reCalc=True)
#
# Exploring the rider
# -------------------
# The bicycle has a few new attributes now that it has a rider:
bicycle.hasRider

# %%
bicycle.riderName

# %%
# inertial parmeters of the rider
pprint.pprint(bicycle.riderPar)

# %%
# this is a yeadon.human object representing the Jason
bicycle.human

# %%
try:
    bicycle.human.print_properties()
except:
    pass

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
_ = bicycle.plot_bicycle_geometry()

# %%
# The Bode plot reflects the changes:
_ = bicycle.plot_bode(3.0, 1, 2)

# %%
# The eigenvalue plot reflects the changes:
_ = bicycle.plot_eigenvalues_vs_speed(speeds)
