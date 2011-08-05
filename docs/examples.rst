=============
Example Usage
==============

Getting bicycle data
====================

Loading bicycle data
====================
The easiest way to load a bicycle is::

  >>>import bicycleparameters as bp
  >>>bicycle = bp.Bicycle('Stratos')

This will create an instance of the Bicycle class in the variable bicycle based
off of input data from the ``./bicycles/Stratos/`` directory. The program first
looks to see if there are any parameter sets in
``./bicycles/Stratos/Parameters/``. If so, it loads the data, if not it looks
for ``./bicycles/Stratos/RawData/StratosMeasurments.txt`` so that it can
generate the parameter set. The raw measurement file may or may not contain the
oscillation period data for the bicycle moment of inertia caluclations. If it
doesn't then the program will look for the series of ``.mat`` files need to
calculate the periods.

There are other loading options::

  >>>bicycle = bp.Bicycle('Stratos', pathToData='<some path to the data directory>', forceRawCalc=Trure, forcePeriodCalc=True)

The ``pathToData`` option allows you specify a directory other than the current
directory as your data directory. The ``forceRawCalc`` forces the program to
load ``./bicycles/Stratos/RawData/StratosMeasurments.txt`` and recalculate the
parameters regarless if there are any parameter files available in
``./bicycles/Stratos/Parameters/``. The ``forcePeriodCalc`` option forces the period
calcluation from the ``.mat`` files regardless if they already exist in the raw
measurement file.

Exploring bicycle data
======================
The benchmark bicycle parameters are the fundamental parameter set that is used
behind the scenes for calculations. To access them type::

  >>>bicycle.parameters['Benchmark']

If the bicycle was calculated from raw data measurements you can access them
by::

  >>>bicycle.parameters['Measurements']

All parameter sets are stored in the parameter dictionary of the bicycle
instance.

To modify a parameter type::

  >>>bicycle.parameters['Benchmark']['mB] = 50.
