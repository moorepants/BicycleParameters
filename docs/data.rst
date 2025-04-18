.. _data-file:

=======================================
BicycleParameters Data File Information
=======================================

The Data
========
The program requires input data in the form of basic text files and a
particular file system directory structure to organize the data. To use the
program you will need to navigate to a directory where you have at least one
directory named for a bicycle that contains parameters files or raw data
measurements. Refer to the document :ref:`data-file` for details about what the
data files should contain.

Data Directory
--------------

You will need to setup a directory (a directory named ``data`` is used in the
following examples) somewhere for the data input and output files. The
structure of the directory should look like this::

    /data
    |
    -->/bicycles
    |  |
    |  -->/Bicyclea
    |  |  |
    |  |  -->/Parameters
    |  |  |
    |  |  -->/Photos
    |  |  |
    |  |  -->/RawData
    |  |
    |  -->/Bicycleb
    |     |
    |     -->/Parameters
    |     |
    |     -->/Photos
    |     |
    |     -->/RawData
    -->/riders
       |
       -->/Ridera
          |
          -->/Parameters
          |
          -->/RawData

Bicycle/rider name
------------------
A bicycle or rider name is a descriptive word (or compound word) for a bicycle or
rider in which the first letter is capitalized. Examples of bicycle short names
include ``Bianchipista``, ``Bike``, ``Mybike``, ``Rigidrider``,
``Schwintandem``, ``Gyrobike``, ``Bicyclea``, etc. Examples of rider names
include ``Jason``, ``Mont``, ``Lukepeterson``, etc. The program relies on
CamelCase words, so make sure the first letter is capitalized and no others
are.

bicycles Directory
------------------
The ``bicycles`` directory contains subdirectories for each bicycle. The
directory name for a bicycle should be its bicycle name. Each directory in
``bicycles`` should contain at least a ``RawData`` directory or a ``Parameters``
directory. ``Photos`` is an optional directory.

RawData directory
~~~~~~~~~~~~~~~~~
You can supply raw measurement data in two forms:

 1. A file containing all the manual measurements (including the oscillation
    periods for each rigid body). Refer to :ref:`bicycle-measured-input` for
    details about the contents of this file.
 2. A file containing all the manual measurements (not including the
    oscillation periods for each rigid body) and a set of data files
    containing oscillatory signals from which the periods can be
    estimated. Refer to :ref:`pendulum-input` for details about these files.

The manual measurement data file should follow the naming convention ``<bicycle
name>Measure.txt``. This data is used to generate parameter files that can be
saved to the ``Parameters`` directory.

Parameters directory
~~~~~~~~~~~~~~~~~~~~
If you don't have any raw measurements for the bicycle it is also an option to
supply a parameter file in the ``Parameters`` directory. Simply add a file named
``<bicycle name>Benchmark.txt`` with the benchmark parameter set into the
``Parameters`` directory for the particular bicycle. Refer to
:ref:`bicycle-parameter-input` for details about the contents of the file.

Photos directory
~~~~~~~~~~~~~~~~
The ``Photos`` folder should contain photos of the bicycle parts hung as the
various pendulums in the various orientations. The filename should follow the
conventions of the raw signal data files.

riders directory
----------------
The riders directory should contain a directory for each rider that you have
data for. The individual rider directory contains a ``Parameters`` for the
rider inertial parameters sets and a ``RawData`` directory for the raw
measurements needed for the Yeadon inertia model. Refer to :ref:`rider-input`
for details about these input files.

Example Data
------------

Example data is available here:

http://dx.doi.org/10.6084/m9.figshare.1198429

.. _bicycle-parameter-input:

bicycles/<bicycle name>/Parameters/<bicycle name>Benchmark.txt
==============================================================
``<bicycle name>Benchmark.txt`` contains the complete parameter set needed to
analyze the Whipple bicycle model linearized about the upright configuration.
Each line should have one of the 24 benchmark parameters in the following
format::

    c = 0.080+/-0.001

The first characters are a unique variable name, followed by an equal sign, the
value of the parameter, a plus or minus symbol (``+/-``), and the standard
deviation of the value. There can be spaces between the parts. Use ``0.0`` for
the standard deviation if this is unknown or if you are not concerned with the
uncertainties. Use the same units as the benchmark bicycle paper. These are the
possible variables:

- ``g`` : acceleration due to gravity [m/s**2]
- ``c`` : trail [m]
- ``w`` : wheelbase [m]
- ``lam`` : steer axis tilt [rad]
- ``rR`` : rear wheel radius [m]
- ``rF`` : front wheel radius [m]
- ``mB`` : frame/rider mass [kg]
- ``mF`` : front wheel mass [kg]
- ``mH`` : handlebar/fork assembly mass [kg]
- ``mR`` : rear wheel mass [kg]
- ``xB`` : x distance to the frame/rider center of mass [m]
- ``zB`` : z distance to the frame/rider center of mass [m]
- ``xH`` : x distance to the frame/rider center of mass [m]
- ``zH`` : z distance to the frame/rider center of mass [m]
- ``IBxx`` : x moment of inertia of the frame/rider [kg*m**2]
- ``IBzz`` : z moment of inertia of the frame/rider [kg*m**2]
- ``IBxz`` : xz product of inertia of the frame/rider [kg*m**2]
- ``IFxx`` : x moment of inertia of the front wheel [kg*m**2]
- ``IFyy`` : y moment of inertia of the front wheel [kg*m**2]
- ``IHxx`` : x moment of inertia of the handlebar/fork [kg*m**2]
- ``IHzz`` : z moment of inertia of the handlebar/fork [kg*m**2]
- ``IHxz`` : xz product of inertia of the handlebar/fork [kg*m**2]
- ``IRxx`` : x moment of inertia of the rear wheel [kg*m**2]
- ``IRyy`` : y moment of inertia of the rear wheel [kg*m**2]

Optional Parameters
-------------------
These parameters are assumed to equal zero if not given.

- ``yB`` : y distance to the frame/rider center of mass [m]
- ``yH`` : y distance to the handlebar/fork center of mass [m]
- ``IBxy`` : xy product of inertia of the frame/rider [kg*m**2]
- ``IByy`` : y moment of inertia of the frame/rider [kg*m**2]
- ``IByz`` : yz product of inertia of the frame/rider [kg*m**2]
- ``IHxy`` : xy product of inertia of the handlebar/fork [kg*m**2]
- ``IHyy`` : y moment of inertia of the handlebar/fork [kg*m**2]
- ``IHyz`` : yz product of inertia of the handlebar/fork [kg*m**2]

.. _bicycle-measured-input:

bicycles/<bicycle name>/RawData/<bicycle name>Measured.txt
==========================================================
This documentation does not contain the complete details of acquiring the raw
data. Please refer to [Moore2012]_ for more information.

``<bicycle name>Measured.txt`` contains the raw measurement data for a bicycle.
The file should have one variable on each line in the following format::

    mR = 1.38+/-0.02, 1.37+/-0.02

This is the same as the previous parameter variable definition accept that
multiple measurements can be included as comma separated values. The values
will be averaged together on import. The following gives the measured values:

- ``aB1`` : perpendicular distance from the pendulum axis to the rear axle
  center, first orienation [m]
- ``aB2`` : perpendicular distance from the pendulum axis to the rear axle
  center, second orienation [m]
- ``aB3`` : perpendicular distance from the pendulum axis to the rear axle
  center, third orienation [m]
- ``aH1`` : perpendicular distance from the pendulum axis to the front axle
  center, first orienation [m]
- ``aH2`` : perpendicular distance from the pendulum axis to the front axle
  center, second orienation [m]
- ``aH3`` : perpendicular distance from the pendulum axis to the front axle
  center, third orienation [m]
- ``alphaB1`` : angle of the head tube with respect to horizontal, first
  orientation [deg]
- ``alphaB2`` : angle of the head tube with respect to horizontal, second
  orientation [deg]
- ``alphaB3`` : angle of the head tube with respect to horizontal, third
  orientation [deg]
- ``alphaH1`` : angle of the steer tube with respect to horizontal, first
  orientation [deg]
- ``alphaH2`` : angle of the steer tube with respect to horizontal, second
  orientation [deg]
- ``alphaH3`` : angle of the steer tube with respect to horizontal, third
  orientation [deg]
- ``dF`` : distance the front wheel travels [m]
- ``dP`` : diameter of the calibration rod [m]
- ``dR`` : distance the rear wheel travels [m]
- ``f`` : fork offset [m]
- ``g`` : acceleration due to gravity [m/s**2]
- ``gamma`` : head tube angle [deg]
- ``lF`` : front wheel compound pendulum length [m]
- ``lP`` : calibration rod length [m]
- ``lR`` : rear wheel compound pendulum length [m]
- ``mB`` : frame mass [kg]
- ``mF`` : front wheel mass [kg]
- ``mH`` : fork/handlebar mass [kg]
- ``mP`` : calibration rod mass [kg]
- ``mR`` : rear wheel mass [kg]
- ``nF`` : number of rotations of the front wheel
- ``nR`` : number of rotations of the rear wheel
- ``TcB1`` : frame compound pendulum oscillation period [s]
- ``TcF1`` : front wheel compound pendulum oscillation period [s]
- ``TcH1`` : fork/handlebar compound pendulum oscillation period [s]
- ``TcR1`` : rear wheel compound pendulum oscillation period [s]
- ``TtB1`` : frame torsional pendulum oscillation period, first orientation [s]
- ``TtB2`` : frame torsional pendulum oscillation period, second orientation [s]
- ``TtB3`` : frame torsional pendulum oscillation period, third orientation [s]
- ``TtF1`` : front wheel torsional pendulum oscillation period, first orientation
  [s]
- ``TtH1`` : handlebar/fork torsional pendulum oscillation period, first
  orientation [s]
- ``TtH2`` : handlebar/fork torsional pendulum oscillation period, second
  orientation [s]
- ``TtH3`` : handlebar/fork torsional pendulum oscillation period, third
  orientation [s]
- ``TtP1`` : calibration torsional pendulum oscillation period [s]
- ``TtR1`` : rear wheel torsional pendulum oscillation period [s]
- ``w`` : wheelbase [m]

Geometry Option
---------------

The default option is to provide the wheelbase ``w``, fork offset ``f``, head
tube angle ``gamma`` and the wheel radii ``rR`` ``rF``, but there is a
secondary option for the geometric variables using the perpendicular distances
from the steer axis to the wheel centers and the distance between their
respective intersection points. To use these, simply replace w, gamma, and f
with these dimensions:

- ``h1`` : distance from the base of the height gage to the top of the rear
  wheel axis [m]
- ``h2`` : distance from the table surface to the base of the height gage [m]
- ``h3`` : distance from the table surface to the top of the head tube [m]
- ``h4`` : height of the top of the front wheel axle [m]
- ``h5`` : height of the top of the steer tube [m]
- ``d1`` : outer diameter of the head tube [m]
- ``d2`` : diameter of the dummy rear axle [m]
- ``d3`` : diameter of of the dummy front axle [m]
- ``d4`` : outer diameter of the steer tube [m]
- ``d`` : inside distance between the rear and the front axles with the fork
  reversed [m]

The details of how to take these measurements can be found in our `raw data
sheet`_ and in [Moore2012]_.

.. _raw data sheet: http://bit.ly/jIeKKB

Rider Configuration Details
---------------------------
A rider can be situated on the bicycle if other raw bicycle measurements are
provided.

- ``lsp`` : the length of the seat post (i.e. the length from the intersection
  of the top tube with the seat tube to the top of the seat along the axis of
  the seat tube. [m]
- ``lst`` : the length of the seat tube (i.e. the distance from the center of
  the bottom bracket to the intersection of the seat tube and the top tube) [m]
- ``hbb`` : the height of the bottom bracket off the ground [m]
- ``lamst`` : the acute angle between horizontal and the seat tube [rad]
- ``lcs`` : the distance from the center of the bottom bracket to the center of
  the rear wheel [m]
- ``LhbR`` : the distance from the center of the rear wheel to either the left
  or right the handlebar grip (roughly where the center of the hand would fall)
  [m]
- ``LhbF`` : the distance from the center of the front wheel to either the left
  or right the handlebar grip (roughly where the center of the hand would fall)
  [m]

Other
-----

You may see these values, ``xcl`` and ``zcl``, in the ``Rigid`` and ``Rigidcl``
bicycles input files. They locate the lateral force point in the benchmark
coordinates. Also, ``ds1`` and ``ds3`` locate the accelerometer with respect to
a point on the steer axis which is aligned with the handlebar center of mass.

Fork/Handlebar Separation
-------------------------
The measurement of the fork and the handlebar as two rigid bodies is also
supported. See the example bicycle called ``Rigid`` for more details. The fork
subscript is ``S`` and the handlebar subscript is ``G``.

Notes
-----

- The periods ``T`` are not required if you provide oscillation signal data
  files.
- You have to specify at least three orientations but more can increase the
  accuracy of the parameter estimations. Currently you can specify up to six
  orientation for each rigid body.

.. _pendulum-input:

Pendulum Data Files
===================

If you have raw signal data that the periods can be estimated from, then these
should be included in the ``RawData`` directory. There should be at least one
file for every period typically found in ``<bicycle name>Measured.txt`` file. The
signals collected should exhibit very typical decayed oscillations. Currently
the only supported file is a Matlab mat file with these variables:

- ``bicycle``, char : short name of the bicycle, e.g. ``'Browser'``
- ``part``, char : examples are ``'Fork'``, ``'Frame'``, ``'Rwheel'``
- ``pendulum``, char : either ``'Compound'`` or ``'Torsional'``
- ``angle`` or ``angleOrder``, char : the ``'First'``, ``'Second'``, ...,
  ``'Sixth'`` orientation
- ``trial``, char : the integer number of the repetition ``'1'``, ``'2'``, etc.
- ``filename``, char : filename constructed from above variables (see below for
  explanation)
- ``notes``, char : any notes recorded for that measurement
- ``data``, double size n x 1 : signal vector of a decaying oscillation (from
  rate gyro)
- Time information, either ``time`` OR both ``duration`` and ``sampleRate``:

   - ``duration``, double size 1 x 1 : time in seconds for the measurement
   - ``sampleRate`` or ``ActualRate``, double size 1 x 1 (integer) : sample
     rate of data in Hertz
   - ``time``, double size n x 1 : time values in seconds that correspond to
     ``data``

The files should be named in this manner ``<short
name><part><pendulum><orientation><trial>.mat`` where:

- ``<bicycle name>`` is the short name of the bicycle
- ``<part>`` is either ``Fork``, ``Handlebar``, ``Frame``, ``Rwheel``, or
  ``Fwheel``
- ``<orientation>`` is either ``First``, ``Second``, ``Third``, ``Fourth``,
  ``Fifth``, or ``Sixth``
- ``<trial>`` is an integer greater than or equal to 1

These data files were originally collected with Matlab and a National
Instruments USB-6218 using a Silicon Sensing CRS03-04S single axis angular rate
gyro and a `data collection script
<https://github.com/moorepants/PhysicalParameters/blob/master/PhysicalParameters/matlab/acquire_data.m>`_.

Notes
-----

- ``Fork`` is the handlebar/fork assembly if they are measured as one rigid body
  (subscript is ``H``). Otherwise ``Fork`` (``S``) is the fork and
  ``Handlebar`` (``G``) is the handlebar when they are measured separately.

.. _rider-input:

riders/<rider name>/Parameters/
===============================

<rider name><bicycle name>Benchmark.txt
---------------------------------------
This file contains the inertial parameters for a rigid rider configured to sit
on a particular bicycle expressed with reference to the benchmark reference
frame and the rider's center of mass. You can provide these values or let the
program generate them.

- ``mB`` : rider mass [kg]
- ``xB`` : x distance to the rider center of mass [m]
- ``yB`` : y distance to the rider center of mass [m]
- ``zB`` : z distance to the rider center of mass [m]
- ``IBxx`` : x moment of inertia of the rider [kg*m**2]
- ``IByy`` : y moment of inertia of the rider [kg*m**2]
- ``IBzz`` : z moment of inertia of the rider [kg*m**2]
- ``IBxy`` : xy product of inertia of the rider [kg*m**2]
- ``IBxz`` : xz product of inertia of the rider [kg*m**2]
- ``IByz`` : yz product of inertia of the rider [kg*m**2]

``yB``, ``IBxy``, and ``IByz`` are optional due to the assumed symmetry of the rider.

Combined/<rider name><bicycle name>Benchmark.txt
------------------------------------------------
This file contains the geometric and inertial benchmark parameters for a rider
seated on a bicycle. The rider is assumed to be rigidly attached to the bicycle
frame. These parameters are the same as the ones stored in
``bicycles/Parameters/<bicycle name>Benchmark.txt``. These file are only output
files.

riders/<rider name>/RawData/
============================

**These files must follow the YAML format used in the yeadon package.**

<rider name><bicycle name>YeadonCFG.txt
---------------------------------------
This is an input file to set the configuration of the joint angles for the
`yeadon package`_. All values should be set to zero except the ``sommersault``
value. The ``sommersault`` value is pi minus the hunch angle of the rider on
the bicycle. The hunch angle is the angle between the horizontal and the
rider's torso mid line. It is essentially the angle at which the rider is
leaned forward.

<rider name><bicycle name>YeadonMeas.txt
----------------------------------------
This is the yeadon measurement input file for the `yeadon package`_. It contains
all of the geometric measurements of the rider. See the `yeadon documentation`_
for more details.

.. _yeadon package: http://pypi.python.org/pypi/yeadon
.. _yeadon documentation : https://yeadon.readthedocs.io
