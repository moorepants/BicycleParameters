Bicycle Parameters Module
=========================

A python program designed to produce and manipulate the basic parameters needed for
the Whipple bicycle model.

History
-------
This is the result of having to measure the physical properties of a bicycle
one two many times.

Features
--------
- Loads bicycle parameter sets from text files
- Generates the benchmark parameters for a real bicycle from experimental data
- Plots a descriptive drawing of the bicycle
- Calculates the eigenvalues for the Whipple bicycle model linearized about the
  upright configuration.
- Plots the eigenvalue root loci as a function of speed, both in the complex
  plane and as eigenvalue vs speed.

Upcoming Features
-----------------
- Converts benchmark parameters to other parametrizations
- Calculates the transfer functions of the open loop system.
- Plots Bode diagrams of the open loop transfer functions.
- Ability to add a rigid rider's physical properties.
- Generates publication quality tables of parameters using LaTeX

Dependencies
------------
These are the versions that I tested the code with, but the code will most
likely work with older versions with minor adjustments.

- [Python 2.6.6](http://www.python.org/)
- [Scipy 0.9.0](http://www.scipy.org/)
- [Numpy 1.5.1](http://numpy.scipy.org/)
- [Matplotlib 0.99.3](http://matplotlib.sourceforge.net/)
- [Uncertainties 1.7.2](http://packages.python.org/uncertainties/)

Installation
------------
For now simply clone the source code with git or download the tarball from
github. Make sure you have the dependencies installed.

Set up your subdirectories like this:

```
/root
|
-->/bicycles
   |
   -->/Shortname
      |
      -->/Parameters
      |
      -->/Photos
      |
      -->/RawData
```
### root directory
The root folder should contain BicycleParameters.py and the other top level
files found in the source code.

### bicycles directory
This directory contains directories for the parameter sets, raw data, and
experiment photos. There should be a folder with a short name for each bicycle
that you have parameter sets and/or raw data for. The short name should be a word
with the first letter capitalized. Examples of Shortname include
"Bianchipista", "Bike", "Mybike", "Rigidrider", "Schwintandem", "Gyrobike", etc.

### Parameters directory
If you don't have any raw measurements for the bicycle, simply add a file
titled `ShortnameBenchmark.txt` with the benchmark parameter set into the
`Parameters` directory for the particular bicycle. Each line should have one of
the 26 benchmark parameters in the following format: `c = 0.080+/-0.001`, where
the first characters are a unique variable name, following with next an equal
sign, the value of the parameter, a plus or minus symbol (`+/-`), and the
standard deviation of the value. There can be spaces between the parts. Use `0.0`
for the standard deviation if this is unknown or you don't need to know the
uncertainties in other values. Use the same units as the benchmark bicycle
paper for less headaches. These are the possible variables:

Required Parameters

- `g` : acceleration due to gravity
- `c` : trail
- `w` : wheelbase
- `lam` : steer axis tilt
- `rR` : rear wheel radius
- `rF` : front wheel radius
- `mB` : frame/rider mass
- `mF` : front wheel mass
- `mH` : handlebar/fork assembly mass
- `mR` : rear wheel mass
- `xB` : x distance to the frame/rider center of mass
- `yB` : y distance to the frame/rider center of mass
- `xH` : x distance to the frame/rider center of mass
- `yH` : y distance to the frame/rider center of mass
- `IBxx` : x moment of inertia of the frame/rider
- `IByy` : y moment of inertia of the frame/rider
- `IBzz` : z moment of inertia of the frame/rider
- `IBxz` : xz product of inertia of the frame/rider
- `IFxx` : x moment of inertia of the front wheel
- `IFyy` : y moment of inertia of the front wheel
- `IHxx` : x moment of inertia of the handlebar/fork
- `IHyy` : y moment of inertia of the handlebar/fork
- `IHzz` : z moment of inertia of the handlebar/fork
- `IHxz` : xz product of inertia of the handlebar/fork
- `IRxx` : x moment of inertia of the rear wheel
- `IRyy` : y moment of inertia of the rear wheel

### RawData directory
If you have raw data it can come in two forms: either a file containing all the
manual measurements (including the oscillation periods for each rigid body) or a file containing
all the manual measurments and a set of data files containing oscillatory
signals from which the periods can be estimated. The manual measurement data
file should follow the naming convention `ShortnameMeasure.txt` and should have
one variable on each line in the following format `mR = 1.38+/-0.02,
1.37+/-0.02` which is the same as the previous parameter variable definition
accept that multiple measurements can be included as comma separated values.

Required Parameters

- aB1 : perpendicular distance from the pendulum axis to the rear axle center, first orienation [m]
- aB2 : perpendicular distance from the pendulum axis to the rear axle center, second orienation [m]
- aB3 : perpendicular distance from the pendulum axis to the rear axle center, third orienation [m]
- aH1 : perpendicular distance from the pendulum axis to the front axle center, first orienation [m]
- aH2 : perpendicular distance from the pendulum axis to the front axle center, second orienation [m]
- aH3 : perpendicular distance from the pendulum axis to the front axle center, third orienation [m]
- alphaB1 : angle of the head tube with respect to horizontal, first orientation [deg]
- alphaB2 : angle of the head tube with respect to horizontal, second orientation [deg]
- alphaB3 : angle of the head tube with respect to horizontal, third orientation [deg]
- alphaH1 : angle of the steer tube with respect to horizontal, first orientation [deg]
- alphaH2 : angle of the steer tube with respect to horizontal, second orientation [deg]
- alphaH3 : angle of the steer tube with respect to horizontal, third orientation [deg]
- dF : distance the front wheel travels [m]
- dP : diameter of the calibration rod [m]
- dR : distance the rear wheel travels [m]
- f : fork offset [m]
- g : acceleration due to gravity [m/s**2]
- gamma : head tube angle [deg]
- lF : front wheel compound pendulum length [m]
- lP : calibration rod length [m]
- lR : rear wheel compound pendulum length [m]
- mB : frame mass [kg]
- mF : front wheel mass [kg]
- mH : fork/handlebar mass [kg]
- mP : calibration rod mass [kg]
- mR : rear wheel mass [kg]
- nF : number of rotations of the front wheel
- nR : number of rotations of the rear wheel
- TcB1 : frame compound pendulum oscillation period [s]
- TcF1 : front wheel compound pendulum oscillation period [s]
- TcH1 : fork/handlebar compound pendulum oscillation period [s]
- TcR1 : rear wheel compound pendulum oscillation period [s]
- TtB1 : frame torsional pendulum oscillation period, first orientation [s]
- TtB2 : frame torsional pendulum oscillation period, second orientation [s]
- TtB3 : frame torsional pendulum oscillation period, third orientation [s]
- TtF1 : front wheel torsional pendulum oscillation period, first orientation [s]
- TtH1 : handlebar/fork torsional pendulum oscillation period, first orientation [s]
- TtH2 : handlebar/fork torsional pendulum oscillation period, second orientation [s]
- TtH3 : handlebar/fork torsional pendulum oscillation period, third orientation [s]
- TtP1 : calibration torsional pendulum oscillation period [s]
- TtR1 : rear wheel torsional pendulum oscillation period [s]
- w : wheelbase [m]

Notes

- The periods (T) are not required if you provide oscillation signal data
  files.

### Photos directory
The Photos folder should contain photos of the bicycle parts hung as the
various pendulums. The

Example Code
------------
```python
import BicycleParameters as bp
rigid = bp.Bicycle('Rigid')
rigid.parameters['Benchmark']
rigid.plot_bicycle_geometry()
speeds = bp.np.linspace(0., 10., num=100)
rigid.plot_eigenvalues_vs_speed(speeds)
```

ToDo
----

- Add the root loci plots.
- Add Bode plots.
- Merge the table generation code.
- Make a bike comparison function.
- Separate the general dynamics functions to another module
