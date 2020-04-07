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
