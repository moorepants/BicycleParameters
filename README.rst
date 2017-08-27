=========================
Bicycle Parameters Module
=========================

A python program designed to produce and manipulate the basic parameters needed
for the Whipple bicycle model.

.. image:: https://img.shields.io/pypi/v/BicycleParameters.svg
   :target: https://pypi.org/project/BicycleParameters/

.. image:: https://anaconda.org/moorepants/bicycleparameters/badges/version.svg
   :target: https://anaconda.org/moorepants/bicycleparameters

.. image:: https://travis-ci.org/moorepants/BicycleParameters.svg?branch=master
   :target: https://travis-ci.org/moorepants/BicycleParameters

Dependencies
============

Required
--------
- `Python 2.7 or >= 3.3 <http://www.python.org/>`_
- `Numpy >= 1.6.1 <http://numpy.scipy.org/>`_
- `Scipy >= 0.9.0 <http://www.scipy.org/>`_
- `Matplotlib >= 1.1.1 <http://matplotlib.sourceforge.net/>`_
- `Uncertainties >= 2.0.0 <http://pypi.python.org/pypi/uncertainties/>`_
- `yeadon >= 1.1.0 <http://pypi.python.org/pypi/yeadon/>`_
- `DynamicistToolKit >= 0.1.0
  <http://pypi.python.org/pypi/DynamicistToolKit>`_

Optional
--------

These are required to build the documentation:

- `Sphinx <http://sphinx.pocoo.org/>`_
- `Numpydoc <http://pypi.python.org/pypi/numpydoc>`_

Installation
============

The easiest method to download and install is with ``pip``::

  $ pip install BicycleParameters

There are other options for getting the source code:

1. Clone the source code with Git: ``git clone
   git://github.com/moorepants/BicycleParameters.git``
2. `Download the source from Github`__.
3. Download the source from pypi__.

.. __: https://github.com/moorepants/BicycleParameters
.. __: http://pypi.python.org/pypi/BicycleParameters

Once you have the source code navigate to the directory and run::

  >>> python setup.py install

This will install the software into your system and you should be able to
import it with::

  >>> import bicycleparameters

Example Code
============

::

    >>> import bicycleparameters as bp
    >>> import numpy as np
    >>> rigid = bp.Bicycle('Rigid')
    >>> par = rigid.parameters['Benchmark']
    >>> rigid.plot_bicycle_geometry()
    >>> speeds = np.linspace(0., 10., num=100)
    >>> rigid.plot_eigenvalues_vs_speed(speeds)

Sample Data
===========

Some sample data is included in the repository but a full source with all the
raw data files can be downloaded from here:

http://dx.doi.org/10.6084/m9.figshare.1198429

Documentation
=============

Please refer to the `online documentation
<http://packages.python.org/BicycleParameters>`_ for more information.

Grant Information
=================

This material is partially based upon work supported by the National Science
Foundation under Grant No. 0928339. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the authors and do not
necessarily reflect the views of the National Science Foundation.

Release Notes
=============

1.0.0
-----

- Support Python 3

0.2.0
-----

- Commands using the state space form of the Whipple model have been reordered
  to [roll angle, steer angle, roll rate, steer rate]
- Added another rider's measurments.
- Added a module for printing tables of data.
- Added the Gyrobike and the ability to manage it's flywheel rigidbody.
- Fixed a bug in `calculate_abc_geometry()` that gave incorrect geometry
  values.
- Handles two additional points for the Davis Instrumented Bicycle.
- Added a child sized person based on scaling Charlie's measurements.
- Added Bode plot commands.
- Added nominal output options for several methods.
- Added a dependency to DynamicistToolKit
- Updated core dependencies to a minimum from the Ubuntu 12.04 release.
- Tested with DTK 0.1.0 to 0.3.5.
- Added Travis support.
- The minimum yeadon version is bumped to 1.1.1 and code updated to reflect the
  new yeadon api.
- The minimum version of uncertainties is bumped to 2.0.

0.1.3
-----

- Speed increase for the eigenvalue calculations.
- Added measurements for the human configuration on some bikes.

0.1.2
-----

- Fixed the tex related bug for the pendulum fit plots
- Fixed some import bugs affecting the split fork/handlebar calcs

0.1.1
-----

- changed the default directory to .
- added pip install notes
- fixed urls in setup.py and the readme
- added version number to the package
- removed the human machine classifier
- reduced the size of the images in the docs
- broke bicycleparameters.py into several modules
- updated the documentation

0.1.0
-----

- Initial release.
