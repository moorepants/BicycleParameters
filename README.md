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
- Generates the benchmark parameter for a real bicycle from experimental data
- Plots a descriptive drawing of the bicycle
- Calculates the eigenvalues for the Whipple bicycle model linearized about the
  upright configuration.
- Plots the eigenvalue root loci as a function of speed, both in the complex
  plane and as eigenvalue vs speed.

Upcoming Features
-----------------
- Converts benchmark parameters to other parametrizations
- Calculates the transfer functions to the system.
- Plots Bode diagrams of the transfer functions
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

Example Code
------------
```python
import BicycleParameters as bp
rigid = bp.Bicycle('Rigid')
rigid.parameters['Benchmark']
rigid.plot_bicycle_geometry()
ridig.plot_eigenvalues_vs_speed()
```
