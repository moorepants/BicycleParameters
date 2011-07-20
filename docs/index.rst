.. BicycleParameters documentation master file, created by
   sphinx-quickstart on Wed Jun 29 15:30:42 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
BicycleParameters's documentation!
==================================

The ``bicycleparameters package`` is a python program designed to produce and
manipulate the basic parameters needed for basic bicycle dynamic models.

Indices and tables
==================

Contents:

.. toctree::
   :maxdepth: 2

   installation.rst
   input.rst

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Description
===========
This is code based off of the work done to measure the physical parameters
of a bicycle and rider at both the `UCD Sports Biomechanics Lab`_ and the `TU
Delft Bicycle Dynamics Lab`_. Physical parameters include but are not limited
to the geometry, mass, mass location and mass distribution of the bicycle rider
system. The code is structured around the Whipple bicycle model and
fundamentally works with and produces the parameters presented in Meijaard
2007, due to the fact that these parameters have been widely adopted as a
benchmark. More detail can be found in our papers and the website_.

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
- Loads a rider from a parameter set.
- Generates the rider parameter set from human measurements based on the Yeadon
  model configured to sit on the bicycle.
- Plots a descriptive drawing of the bicycle or rider.

Basic Linear Analysis
---------------------

- Calculates the A and B matrices for the Whipple bicycle model linearized
  about the upright configuration.
- Calculates the canonical matrices for the Whipple bicycle model linearized
  about the upright configuration.
- Calculates the eigenvalues for the Whipple bicycle model linearized about the
  upright configuration.
- Plots the eigenvalue root loci as a function of speed, both in the complex
  plane and as eigenvalue vs speed.

Upcoming Features
=================

- Converts benchmark parameters to other parametrizations
- Calculates the transfer functions of the open loop system.
- Plots Bode diagrams of the open loop transfer functions.
- Generates publication quality tables of parameters using LaTeX

References
==========
The methods associated with this software were built on these previous works,
among others.

1. Moore, J. K., Hubbard, M., Peterson, D. L., Schwab, A. L., and Kooijman, J.
   D. G. (2010). An accurate method of measuring and comparing a bicycle's
   physical parameters. In Bicycle and Motorcycle Dynamics: Symposium on the
   Dynamics and Control of Single Track Vehicles, Delft, Netherlands.
2. Moore, J. K., Kooijman, J. D. G., Hubbard, M., and Schwab, A. L. (2009). A
   Method for Estimating Physical Properties of a Combined Bicycle and Rider.
   In Proceedings of the ASME 2009 International Design Engineering Technical
   Conferences & Computers and Information in Engineering Conference,
   IDETC/CIE 2009, San Diego, CA, USA. ASME.
3. Kooijman, J. D. G., Schwab, A. L., and Meijaard, J. P. (2008). Experimental
   validation of a model of an uncontrolled bicycle. Multibody System Dynamics,
   19:115–132.
4. Kooijman, J. D. G. (2006). Experimental validation of a model for the motion
   of an uncontrolled bicycle. MSc thesis, Delft University of Technology.
5. Roland J R ., R. D., and Massing , D. E. A digital computer simulation of
   bicycle dynamics. Calspan Report YA-3063-K-1, Cornell Aeronautical
   Laboratory, Inc., Buffalo, NY, 14221, Jun 1971. Prepared for Schwinn Bicycle
   Company, Chicago, IL 60639.
