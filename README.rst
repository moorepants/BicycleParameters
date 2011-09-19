=========================
Bicycle Parameters Module
=========================

A python program designed to produce and manipulate the basic parameters needed for
the Whipple bicycle model.

Dependencies
============
These are the versions that I tested the code with, but the code will most
likely work with older versions.

Required
--------
- `Python 2.7.1 <http://www.python.org/>`_
- `Scipy 0.9.0 <http://www.scipy.org/>`_
- `Numpy 1.5.1 <http://numpy.scipy.org/>`_
- `Matplotlib 0.99.3 <http://matplotlib.sourceforge.net/>`_
- `Uncertainties 1.7.3 <http://pypi.python.org/pypi/uncertainties/>`_
- `yeadon 0.8 <http://pypi.python.org/pypi/yeadon/>`_

Optional
--------
These are required to bulid the documentation:

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
3. Dowload the source from pypi__.

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

Documentation
=============
Please refer to the `online documentation
<http://packages.python.org/BicycleParameters>`_ for more information.

Release Notes
=============

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
Initial release.
