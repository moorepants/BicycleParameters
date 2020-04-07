=================
BicycleParameters
=================

A Python program designed to generate, manipulate, and visualize the parameters
of the Whipple-Carvallo bicycle model.

.. list-table::

   * - Download from PyPi
     - |PyPi|
   * - Download from Anaconda
     - |Anaconda|
   * - Documentation
     - http://packages.python.org/BicycleParameters
   * - Travis CI Status
     - |Travis|

.. |PyPi| image:: https://img.shields.io/pypi/v/BicycleParameters.svg
   :target: https://pypi.org/project/BicycleParameters/

.. |Anaconda| image:: https://anaconda.org/conda-forge/bicycleparameters/badges/version.svg
   :target: https://anaconda.org/conda-forge/bicycleparameters

.. |Travis| image:: https://travis-ci.org/moorepants/BicycleParameters.svg?branch=master
   :target: https://travis-ci.org/moorepants/BicycleParameters

Dependencies
============

Required
--------

- `Python 2.7 or >= 3.5 <http://www.python.org/>`_
- `Numpy >= 1.6.1 <https://numpy.org/>`_
- `Scipy >= 0.9.0 <https://scipy.org/>`_
- `Matplotlib >= 1.1.1 <https://matplotlib.org/>`_
- `Uncertainties >= 2.0.0 <https://pythonhosted.org/uncertainties/>`_
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

We recommend installing BicycleParameters with conda_ or pip_.

.. _conda: https://docs.conda.io
.. _pip: https://pip.pypa.io

For conda::

  $ conda install -c conda-forge bicycleparameters

For pip::

  $ pip install BicycleParameters

The package can also be installed from the source code. The options for getting
the source code are:

1. Clone the source code with Git: ``git clone
   git://github.com/moorepants/BicycleParameters.git``
2. `Download the source from Github`__.
3. Download the source from pypi__.

.. __: https://github.com/moorepants/BicycleParameters
.. __: http://pypi.python.org/pypi/BicycleParameters

Once you have the source code navigate to the directory and run::

  >>> python setup.py install

This will install the software into your system. You can check if it installs
with::

   $ python -c "import bicycleparameters"

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
Foundation under `Grant No. 0928339`_. Any opinions, findings, and conclusions
or recommendations expressed in this material are those of the authors and do
not necessarily reflect the views of the National Science Foundation.

.. _Grant No. 0928339: https://www.nsf.gov/awardsearch/showAward?AWD_ID=0928339
