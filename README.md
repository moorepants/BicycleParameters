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
titled ShortnameBenchmark.txt with the benchmark parameter set into the
Parameters directory for the particular bicycle. Each line should have one of
the 26 benchmark parameters in the following format:

c = 0.080+/-0.001

Where the first characters are a unique variable name, following with next an
equal sign, the value of the parameter, a plus or minus symbol ('+/-'), and the
standard deviation of the value. There can be spaces between the parts. Use 0.0
for the standard deviation if this is unknown or you don't need to know the
uncertainties in other values. Use the same units as the benchmark bicycle
paper for less headaches. These are the possible variables:

Required Parameters
- g : acceleration due to gravity
- c : trail
- w : wheelbase
- lam : steer axis tilt
- rR : rear wheel radius
- rF : front wheel radius
- mB : frame/rider mass
- mF : front wheel mass
- mH : handlebar/fork assembly mass
- mR : rear wheel mass
- xB : x distance to the frame/rider center of mass
- yB : y distance to the frame/rider center of mass
- xH : x distance to the frame/rider center of mass
- yH : y distance to the frame/rider center of mass
- IBxx : x moment of inertia of the frame/rider
- IByy : y moment of inertia of the frame/rider
- IBzz : z moment of inertia of the frame/rider
- IBxz : xz product of inertia of the frame/rider
- IFxx : x moment of inertia of the front wheel
- IFyy : y moment of inertia of the front wheel
- IHxx : x moment of inertia of the handlebar/fork
- IHyy : y moment of inertia of the handlebar/fork
- IHzz : z moment of inertia of the handlebar/fork
- IHxz : xz product of inertia of the handlebar/fork
- IRxx : x moment of inertia of the rear wheel
- IRyy : y moment of inertia of the rear wheel

### RawData directory
If you have raw data, the RawData folder should contain a file named
ShortnameMeasure.txt file that contains all of the manually obtained raw data
and potentially the mat files with the rate gyro data from the torsional and
compound pendulums measurements.

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
