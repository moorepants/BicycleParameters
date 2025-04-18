============
Installation
============

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
