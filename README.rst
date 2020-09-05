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
   * - Heroku App
     - |heroku|

.. |PyPi| image:: https://img.shields.io/pypi/v/BicycleParameters.svg
   :target: https://pypi.org/project/BicycleParameters/

.. |Anaconda| image:: https://anaconda.org/conda-forge/bicycleparameters/badges/version.svg
   :target: https://anaconda.org/conda-forge/bicycleparameters

.. |Travis| image:: https://travis-ci.org/moorepants/BicycleParameters.svg?branch=master
   :target: https://travis-ci.org/moorepants/BicycleParameters

.. |heroku| image:: http://heroku-badge.herokuapp.com/?app=bicycleparameters&svg=1
   :target: https://bicycleparameters.herokuapp.com
   :alt: Heroku Application

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


=============================
Bicycle Dynamics Analysis App
=============================

The Bicycle Dynamics Analysis App provides a GUI for using the
BicycleParameters Python program on the web. Using Dash, we transform
Python code into HTML and CSS which is nicely rendered within a web
browser. The application is available at
https://bicycleparameters.herokuapp.com/.

Directory Structure
===================

Within the ``BicycleParameters/bicycleparameters/`` directory, the
primary module for the application is ``cycle_app.py`` and it is written
entirely in Python. The ``/assets/`` folder contains files which are to
be displayed by the app, and the ``/data/`` folder contains raw bicycle
measurement data. Some custom CSS is contained within
``/assets/styles.css``, but most of the CSS can be written directly in
``cycle_app.py`` using ``dash-bootstrap-components``. You can read more
about the purpose of the ``/assets/`` folder on the `Dash
documentation <https://dash.plotly.com/external-resources>`__.

Development
===========

Developing the application is quite simple. First, you need to have an
environment which contains all the necessary dependencies for running
``cycle_app.py``. If you are using ``conda``, you can use the
``environment.yml`` located in the top-level ``BicycleParameters/``
directory to `build a conda
environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`__
which contains all the packages you need to develop and run this code.
Alternatively, you can simply view the contents of ``environment.yml``
and install those packages accordingly or use the ``requirements.txt``
file available in the same location.

To run ``cycle_app.py`` and view the contents locally, navigate to the
``BicycleParameters/`` directory and run

::

    python -m bicycleparameters.cycle_app

This command calls python and passes the file
``bicycleparameters.cycle_app`` (this is like
``/bicycleparameters/cycle_app.py``) using the ``-m`` flag as if we were
simply running a python script. If you instead navigate to
``/bicycleparameters/`` and run ``python cycle_app.py``, the plot images
will not appear in your browser. Stick to the first command above.

If ``cycle_app.py`` has been executed properly, you should see an output
similar to the following;

::

    (bp) user@host:~/BicycleParameters/bicycleparameters$ python -m bicycleparameters.cycle_app
    Dash is running on http://127.0.0.1:8050/

     Warning: This is a development server. Do not use app.run_server
     in production, use a production WSGI server like gunicorn instead.

     * Serving Flask app "cycle_app" (lazy loading)
     * Environment: production
       WARNING: This is a development server. Do not use it in a production deployment.
       Use a production WSGI server instead.
     * Debug mode: on

Now if you navigate to http://127.0.0.1:8050/, you should see your local
version of the app displayed in your browser. Congratulations! As you
play with the application online you should see feedback within your
terminal window. Debug information will also display here. In addition,
I recommend using the inspect element tool available with most browsers
to debug things live within your browser.

Additional Resources
====================

Here are some resources that I found very useful when first developing
this application:

-  The offical `Dash documentation <https://dash.plotly.com/>`__. Just
   about every single link on this page will have useful information for
   you.
-  `Dash Bootstrap Components
   documentation <https://dash-bootstrap-components.opensource.faculty.ai/docs/components/>`__.
   This is used to write `CSS
   Bootstraps <https://getbootstrap.com/docs/3.3/css/>`__ using the
   Python language.
-  The `Mozilla Web Development
   guide <https://developer.mozilla.org/en-US/docs/Learn>`__. I highly
   recommened this guide for learning about HTML, CSS, and general web
   development.
-  The `example
   usage <https://pythonhosted.org/BicycleParameters/examples.html>`__
   page for Bicycle Parameters. Useful for understanding how the backend
   code works.
-  `w3schools.com <https://www.w3schools.com/>`__. Has great HTML/CSS
   reference pages as well as tutorials. Also has some for
   `Python <https://www.w3schools.com/python/default.asp>`__.
-  This `Software
   Carpentery <https://carpentries.github.io/workshop-template/>`__
   site. Has nice general programming tutorials as well as an in-depth
   `git
   tutorial <https://swcarpentry.github.io/git-novice/reference>`__.

Feel free to extend this list as you develop and learn. Overall, I ended
up needing to learn and use web development skills far more than I
needed to really understand Python itself. Program in whichever way
brings you the most joy. I wish you the best, future devs!

