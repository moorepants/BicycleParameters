===========
Description
===========

This is code based off of the work done to measure the physical parameters of a
bicycle and rider at both the `UCD Sports Biomechanics Lab`_ and the `TU Delft
Bicycle Dynamics Lab`_. Physical parameters include but are not limited to the
geometry, mass, mass location and mass distribution of the bicycle rider
system. The code is structured around the Whipple bicycle model and
fundamentally works with and produces the parameters presented in Meijaard 2007
[Meijaard2007]_, due to the fact that these parameters have been widely adopted
as a benchmark. But the software is also capable of generating parameter sets
for more complex rider biomechanical models. More detail can be found in our
papers and the website_ and in :ref:`references`.

.. _UCD Sports Biomechanics Lab: http://biosport.ucdavis.edu
.. _TU Delft Bicycle Dynamics Lab: http://bicycle.tudelft.edu
.. _website: http://biosport.ucdavis.edu/research-projects/bicycle/bicycle-parameter-measurement

Features
========

Parameter Manipulation
----------------------

- Loads bicycle parameter sets from a text file into a python object.
- Generates the benchmark parameter set for a real bicycle from experimental
  data.
- Generates the rider parameter set from human measurements based on the Yeadon
  model configured to sit on the bicycle.
- Plots a descriptive drawing of the bicycle and/or rider.
- Generates publication quality tables of parameters.

Basic Linear Analysis
---------------------

- Calculates the A and B matrices for the Whipple bicycle model linearized
  about the upright configuration.
- Calculates the canonical matrices for the Whipple bicycle model linearized
  about the upright configuration.
- Calculates the eigenvalues for the Whipple bicycle model linearized about the
  upright configuration.
- Plots the eigenvalue root loci as a function of speed as eigenvalue vs speed.
- Plots Bode diagrams of the open loop transfer functions.

Refer to :ref:`usage` for examples of the features.

Upcoming Features
=================

- Converts benchmark parameters to other parametrizations.
- Calculates the transfer functions of the open loop system.

Example Code
============

::

    >>> import bicycleparameters as bp
    >>> import numpy as np
    >>> rigid = bp.Bicycle('Rigid')
    >>> par = rigid.parameters['Benchmark']
    >>> rigid.plot_bicycle_geometry()
    >>> speeds = np.linspace(0., 10., num=100)
    >>> rigid.plot_eigenvalues_vs_speed(speeds, show=True)

.. _references:

References
==========

The methods associated with this software were built upon these previous works,
among others.

.. [Carvallo1899] Carvallo, E. (1899). Théorie du mouvement du monocycle et de
   la bicyclette. Gauthier- Villars.

.. [Whipple1899] Whipple, F. J. W. (1899). The stability of the motion of a
   bicycle. Quarterly Journal of Pure and Applied Mathematics, 30, 312–348.

.. [Roland1971] Roland J R ., R. D., and Massing , D. E. A digital computer simulation of
   bicycle dynamics. Calspan Report YA-3063-K-1, Cornell Aeronautical
   Laboratory, Inc., Buffalo, NY, 14221, Jun 1971. Prepared for Schwinn Bicycle
   Company, Chicago, IL 60639.

.. [Meijaard2007] Meijaard, J. P.; Papadopoulos, J. M.; Ruina, A. & Schwab, A.
   L. Linearized dynamics equations for the balance and steer of a bicycle: A
   benchmark and review Proceedings of the Royal Society A: Mathematical, Physical
   and Engineering Sciences, 2007, 463, 1955-1982

.. [Kooijman2006] Kooijman, J. D. G. (2006). Experimental validation of a model for the motion
   of an uncontrolled bicycle. MSc thesis, Delft University of Technology.

.. [Kooijman2008] Kooijman, J. D. G., Schwab, A. L., and Meijaard, J. P. (2008). Experimental
   validation of a model of an uncontrolled bicycle. Multibody System Dynamics,
   19:115–132.

.. [Moore2009] Moore, J. K., Kooijman, J. D. G., Hubbard, M., and Schwab, A. L. (2009). A
   Method for Estimating Physical Properties of a Combined Bicycle and Rider.
   In Proceedings of the ASME 2009 International Design Engineering Technical
   Conferences & Computers and Information in Engineering Conference,
   IDETC/CIE 2009, San Diego, CA, USA. ASME.

.. [Moore2010] Moore, J. K., Hubbard, M., Peterson, D. L., Schwab, A. L., and Kooijman, J.
   D. G. (2010). An accurate method of measuring and comparing a bicycle's
   physical parameters. In Bicycle and Motorcycle Dynamics: Symposium on the
   Dynamics and Control of Single Track Vehicles, Delft, Netherlands.

.. [Moore2012] Moore, J. K. (2012). Human Control of a Bicycle. University of
   California, Davis PhD Thesis. http://moorepants.github.io/dissertation

.. [Dembia2014] Dembia C, Moore JK and Hubbard M. An object oriented
   implementation of the Yeadon human inertia model [v1; ref status: awaiting
   peer review, http://f1000r.es/4cr] F1000Research 2014, 3:223 (doi:
   10.12688/f1000research.5292.1)
