=================
BicycleParameters
=================

A Python program designed to generate, manipulate, and visualize the parameters
of the bicycle dynamics models.

.. list-table::

   * - Download from PyPi
     - |PyPi|
   * - Download from Anaconda
     - |Anaconda|
   * - Documentation
     - |RTD|
   * - CI Status
     - |GHCI|
   * - Render App
     - |Render|

.. |PyPi| image:: https://img.shields.io/pypi/v/BicycleParameters.svg
   :target: https://pypi.org/project/BicycleParameters/

.. |Anaconda| image:: https://anaconda.org/conda-forge/bicycleparameters/badges/version.svg
   :target: https://anaconda.org/conda-forge/bicycleparameters

.. |GHCI| image:: https://github.com/moorepants/BicycleParameters/actions/workflows/test.yml/badge.svg

.. |RTD| image:: https://readthedocs.org/projects/bicycleparameters/badge/?version=latest
   :target: https://bicycleparameters.readthedocs.io/
   :alt: Documentation Status

.. |Render| image:: https://img.shields.io/badge/Bicycle_Dynamics_App-Render.io-blue
   :target: https://bicycle-dynamics.onrender.com

Dependencies
============

Required
--------

- `DynamicistToolKit >= 0.5.3 <http://pypi.python.org/pypi/DynamicistToolKit>`_
- `Matplotlib >= 3.5.1 <https://matplotlib.org/>`_
- `NumPy >= 1.21.5 <https://numpy.org/>`_
- `Python >= 3.8 <http://www.python.org/>`_
- `SciPy >= 1.8.0 <https://scipy.org/>`_
- `Uncertainties >= 3.1.5 <https://pythonhosted.org/uncertainties/>`_
- `yeadon >= 1.3.0 <http://pypi.python.org/pypi/yeadon/>`_

Optional
--------

These are required to run the Dash web application:

- `Dash >= 2.0 <https://plotly.com/dash/>`_
- `dash-bootstrap-components <https://github.com/facultyai/dash-bootstrap-components>`_
- `Pandas >= 1.3.5 <https://pandas.pydata.org/>`_

These are required to build the documentation:

- `Sphinx >= 4.3.2 <http://sphinx.pocoo.org/>`_
- `Numpydoc >= 1.2 <http://pypi.python.org/pypi/numpydoc>`_
- `sphinx-reredirects <https://documatt.com/sphinx-reredirects/>`_
- `Sphinx-Gallery <https://sphinx-gallery.github.io/stable/index.html>`_

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

.. code-block:: python

    >>> import bicycleparameters as bp
    >>> import numpy as np
    >>> rigid = bp.Bicycle('Rigid')
    >>> par = rigid.parameters['Benchmark']
    >>> rigid.plot_bicycle_geometry()
    >>> speeds = np.linspace(0., 10., num=100)
    >>> rigid.plot_eigenvalues_vs_speed(speeds)

See `example usage <https://bicycleparameters.readthedocs.io/stable/examples.html>`_ in the
documentation.

Sample Data
===========

Some sample data is included in the repository but a full source with all the
raw data files can be downloaded from here:

http://dx.doi.org/10.6084/m9.figshare.1198429

Documentation
=============

Please refer to the `online documentation
<https://bicycleparameters.readthedocs.io/>`_ for more information.

Grant Information
=================

This material is partially based upon work supported by the National Science
Foundation under `Grant No. 0928339`_. Any opinions, findings, and conclusions
or recommendations expressed in this material are those of the authors and do
not necessarily reflect the views of the National Science Foundation.

.. _Grant No. 0928339: https://www.nsf.gov/awardsearch/showAward?AWD_ID=0928339

This material is partially based upon work supported by the TKI CLICKNL grant
"Fiets van de Toekomst"(Grant No. TKI1706).
