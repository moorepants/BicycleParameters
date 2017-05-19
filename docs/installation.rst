==============================
Bicycleparameters Installation
==============================

Dependencies
============
These are the versions that I tested the code with, but the code will most
likely work with older versions.

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
