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
calculate the periods. If no data is there, then you get an error.

There are other loading options::

  >>>bicycle = bp.Bicycle('Stratos', pathToData='<some path to the data directory>', forceRawCalc=Trure, forcePeriodCalc=True)

The ``pathToData`` option allows you specify a directory other than the current
directory as your data directory. The ``forceRawCalc`` forces the program to
load ``./bicycles/Stratos/RawData/StratosMeasurments.txt`` and recalculate the
parameters regarless if there are any parameter files available in
``./bicycles/Stratos/Parameters/``. The ``forcePeriodCalc`` option forces the period
calcluation from the ``.mat`` files regardless if they already exist in the raw
measurement file.

Exploring bicycle parameter data
================================

The bicycle has a name::

  >>>bicycle.bicycleName
  'Stratos'

and a directory where its data is stored::

  >>>bicycle.direcotory
  './bicycles/Stratos'

The benchmark bicycle parameters are the fundamental parameter set that is used
behind the scenes for calculations. To access them type::

  >>>bPar = bicycle.parameters['Benchmark']
  >>>bPar['xB']
  0.32631503794489763+/-0.0032538862692938642

If the bicycle was calculated from raw data measurements you can access them
by::

  >>>bicycle.parameters['Measurements']

All parameter sets are stored in the parameter dictionary of the bicycle
instance.

To modify a parameter type::

  >>>bicycle.parameters['Benchmark']['mB] = 50.

You can regenerate the parameter sets from the raw data stored in the bicycle's
directory by calling::

  >>>bicycle.calculate_from_measured()

Basic Analysis
==============
The program has some basic bicycle analysis tools based on the Whipple bicycle
model which has been linearized about the upright configuration. You can
calculate the eigenvalues and eigenvectors at any speed by calling::

   >>>w, v = bicycle.eig(4.28) # the speed should be in meters/second
   >>>w # eigenvalues
   array([[-6.83490195+0.j        ,  0.46085314+2.77336727j,
            0.46085314-2.77336727j, -1.58257375+0.j        ]])
   >>>v # eigenvectors
   array([[[ 0.04283049+0.j        ,  0.50596715+0.33334818j,
             0.50596715-0.33334818j,  0.55478588+0.j        ],
           [ 0.98853840+0.j        ,  0.72150298+0.j        ,
             0.72150298+0.j        ,  0.63786241+0.j        ],
           [-0.00626644+0.j        ,  0.14646768-0.15809917j,
             0.14646768+0.15809917j, -0.35055926+0.j        ],
           [-0.14463096+0.j        ,  0.04206844-0.25316359j,
             0.04206844+0.25316359j, -0.40305383+0.j        ]]])

Plots
-----
You can
