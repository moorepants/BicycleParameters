===================================
BicycleParameters Input Information
===================================

Parameters/<short name>Benchmark.txt
====================================
``<short name>Benchmark.txt`` contains the complete parameter set needed to
analyze the Whipple bicycle model linearized about the upright configuration.
Each line should have one of the 24 benchmark parameters in the following
format::
    c = 0.080+/-0.001
The first characters are a unique variable name, followed by an equal sign, the
value of the parameter, a plus or minus symbol (``+/-``), and the standard
deviation of the value. There can be spaces between the parts. Use ``0.0`` for
the standard deviation if this is unknown or you don't need to know the
uncertainties in other values. Use the same units as the benchmark bicycle
paper for less headaches. These are the possible variables:

- ``g`` : acceleration due to gravity
- ``c`` : trail
- ``w`` : wheelbase
- ``lam`` : steer axis tilt
- ``rR`` : rear wheel radius
- ``rF`` : front wheel radius
- ``mB`` : frame/rider mass
- ``mF`` : front wheel mass
- ``mH`` : handlebar/fork assembly mass
- ``mR`` : rear wheel mass
- ``xB`` : x distance to the frame/rider center of mass
- ``zB`` : z distance to the frame/rider center of mass
- ``xH`` : x distance to the frame/rider center of mass
- ``zH`` : z distance to the frame/rider center of mass
- ``IBxx`` : x moment of inertia of the frame/rider
- ``IBzz`` : z moment of inertia of the frame/rider
- ``IBxz`` : xz product of inertia of the frame/rider
- ``IFxx`` : x moment of inertia of the front wheel
- ``IFyy`` : y moment of inertia of the front wheel
- ``IHxx`` : x moment of inertia of the handlebar/fork
- ``IHzz`` : z moment of inertia of the handlebar/fork
- ``IHxz`` : xz product of inertia of the handlebar/fork
- ``IRxx`` : x moment of inertia of the rear wheel
- ``IRyy`` : y moment of inertia of the rear wheel

Optional Parameters
-------------------
These parameters are assumed to equal zero if not given.

- ``yB`` : y distance to the frame/rider center of mass
- ``yH`` : y distance to the handlebar/fork center of mass
- ``IBxy`` : xy product of inertia of the frame/rider
- ``IByy`` : y moment of inertia of the frame/rider
- ``IByz`` : yz product of inertia of the frame/rider
- ``IHxy`` : xy product of inertia of the handlebar/fork
- ``IHyy`` : y moment of inertia of the handlebar/fork
- ``IHyz`` : yz product of inertia of the handlebar/fork

RawData/<short name>Measured.txt
=============================
``<Shortname>Measured.txt`` contains the raw measurement data for a bicycle.
The file should have one variable on each line in the following format::
    mR = 1.38+/-0.02, 1.37+/-0.02
This is the same as the previous parameter variable
definition accept that multiple measurements can be included as comma separated
values. The values will be averaged together on import. The following gives the
measured values:

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

- ``h1`` : distance from the base of the height gage to the top of the the rear
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
sheet`_ and on our website_.

.. _raw data sheet: http://bit.ly/jIeKKB
.. _website: http://biosport.ucdavis.edu/research-projects/bicycle/bicycle-parameter-measurement/frame-dimensions

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

Pendulum Data Files
===================
If you have raw signal data that the periods can be estimated from, then these
should be included in the ``RawData`` directory. There should be at least one
file for every period typically found in ``ShortnameMeasured.txt`` file. The
signals collected should exhibit very typical decayed oscillations. Currently
the only supported file is a Matlab mat file with these variables:

- ``data`` : signal vector of a decaying oscillation
- ``sampleRate`` : sample rate of data in hertz

The files should be named in this manner ``<short
name><part><pendulum><orientation><trial>.mat`` where:

- ``<short name>`` is the short name of the bicycle
- ``<part>`` is either ``Fork``, ``Handlebar``, ``Frame``, ``Rwheel``, or
  ``Fwheel``
- ``<orientation>`` is either ``First``, ``Second``, ``Third``, ``Fourth``,
  ``Fifth``, or ``Sixth``
- ``<trial>`` is an integer greater than or equal to 1

Notes
-----

- ``Fork`` is the handlbar/fork assembly if they are measured as one rigid body
  (subscript is ``H``). Otherwise ``Fork`` (``S``) is the fork and
  ``Handlebar`` (``G``) is the handlebar when they are measured separately.
