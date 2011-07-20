==============================
Bicycleparameters Installation
==============================

Installation
============
There are currently two options for getting the source code:

1. Clone the source code with Git: `$git clone
   git://github.com/moorepants/BicycleParameters.git`
2. `Download the source from Github`__.

.. __: https://github.com/moorepants/BicycleParameters

Dependencies
============
These are the versions that I tested the code with, but the code will most
likely work with older versions.

- `Python 2.7.1`_
- `Scipy 0.9.0`_
- `Numpy 1.5.1`_
- `Matplotlib 0.99.3`_
- `Uncertainties 1.7.3`_
- `yeadon 0.8`_

.. _Python 2.7.1: http://www.python.org/
.. _Numpy 1.5.1: http://numpy.scipy.org/
.. _Scipy 0.9.0: http://www.scipy.org/
.. _Uncertainties 1.7.3: http://packages.python.org/uncertainties/
.. _Matplotlib 0.99.3: http://matplotlib.sourceforge.net/
.. _yeadon 0.8: https://github.com/fitze/yeadon

data Directory
==============

You will need to setup a data directory somewhere for the data input and output
files. The structure of the directory should look like this::

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

Short name
==========
A short name is a descriptive word (or compound word) for a bicycle or rider in
which the first letter is capitalized. Examples of bicycle short names include
`Bianchipista`, `Bike`, `Mybike`, `Rigidrider`, `Schwintandem`, `Gyrobike`,
`Bicyclea`, etc. Examples of rider short names include `Jason`, `Mont`,
`Lukepeterson`, etc. The program relies on CamelCase words, so make sure the
first letter is capitalized and no others are.

bicycles Directory
==================
The `bicycles` directory contains subdirectories for each bicycle. The
directory name for a bicycle should be its short name. Each directory in
`bicycles` should contain at least a `RawData` directory or a `Parameters`
directory. `Photos` is an optional directory.

RawData directory
-----------------
You can supply raw measurement data two forms:

 1. A file containing all the manual measurements (including the oscillation
    periods for each rigid body)
 2. A file containing all the manual measurements (not including the
    oscialloation periods for each rigid body) and a set of data files
    containing oscillatory signals from which the periods can be estimated.

The manual measurement data file should follow the naming convention `<short
name>Measure.txt`. This data is used to generate parameter files in the
`Parameters` directory.

Parameters directory
--------------------
If you don't have any raw measurements for the bicycle it is also an option to
supply a parameter file in the `Parameters` direcotry. Simply add a file named
`<short name>Benchmark.txt` with the benchmark parameter set into the
`Parameters` directory for the particular bicycle.

Photos directory
----------------
The `Photos` folder should contain photos of the bicycle parts hung as the
various pendulums in the various orientations. The filename should follow the
conventions of the raw signal data files.
