# Introduction

This web application computes the eigenvalues of the Whipple-Carvallo [1,2]
linearized bicycle model about the upright constant speed equilibrium state.
The eigenvalues are a function of 27 parameters as defined in [3]. The
eigenvalues can be used to assess the uncontrolled stability ("self-stability")
of a given bicycle design as well as characterize the growth and decay rates
and oscillatory nature of the time evolutions of the two degrees of freedom:
roll and steer.

# Inputs

## Parameter Set Selection

Several pre-defined bicycles are provided and can be selected for analysis.
These bicycles and the methods of obtaining the parameter values are described
in detail in [4,5].

- Benchmark: Benchmark parameter set in [3]. Note this includes the rider.
- Batavus Browser: Urban bicycle with swept back handlebars
- Rigid: Instrumented bicycle used in [4]
- Batavus Crescendo Deluxe: Urban bicycle with swept back handlebars
- Batavus Stratos Deluxe: Urban bicycle with straight handlebars
- Gary Fisher: Mountain bike
- Bianchi Pista: Track bicycle
- Silver: Bicycle used in [6]
- Yellow Bicycle: Simple road bicycle with minimal accessories
- Yellow Bicycle Reversed Fork: Simple road bicycle with minimal accessories
  with the fork rotated 180 degrees from the normal orientation.

## Speed Range Selector

Use the slider to select the range of speeds the eigenvalues are calculated
for. This corresponds to the abscissa of the "Eigenvalues vs. Speed" plot.

## Model Parameters

The numerical values of 27 parameters can be edited in the table. These
parameters are defined in [3]. Select the numerical value, change the number to
the desired number, and press enter to see the figures update. Note the units
of the parameters. The "Reset Table" button will change all of the values in
the table back to the currently selected bicycle parameter set.

# Outputs

## Geometry Plot

This plot depicts a side view schematic of the basic bicycle geometry: circles
representing the front and rear wheels and lines representing the steer axis
(dotted) and front and rear wheel offsets from the steer axis. The mass centers
of the front wheel [F], rear wheel [R], rear frame (and rider) [B], and fork
and handlebar [H] are indicated for each rigid body. Inertial ellipsoids are
shown for each body with a matching color to the mass center symbol.

## Eigenvalue Plot

The real parts (solid lines) and imaginary parts (dotted lines) of the
eigenvalues are plotted on the ordinate versus speed on the abscissa. Any
speeds in which the real parts are all negative indicate self-stability at
those speeds. The "Show Self-stability" checkbox will plot a shaded region for
these speeds.

# Colophon

This website was designed by Lyla Sanders and Julie van Vlerken with mentorship
from Jason K. Moore and Andrew Dressel.

The software that powers the website is open source and information on it can
be found here:

- Github Repository: https://github.com/moorepants/BicycleParameters
- Documentation: https://bicycleparameters.readthedocs.io

Contributions and issue reports are welcome!

This software is partially based upon work supported by the National Science
Foundation under [Grant No.
0928339](https://www.nsf.gov/awardsearch/showAward?AWD_ID=0928339). Any
opinions, findings, and conclusions or recommendations expressed in this
material are those of the authors and do not necessarily reflect the views of
the National Science Foundation.

This software is partially based upon work supported by the TKI CLICKNL grant
"Fiets van de Toekomst"(Grant No. TKI1706).

# References

1. F. J. W. Whipple, "The stability of the motion of a bicycle," Quarterly
   Journal of Pure and Applied Mathematics, vol. 30, pp. 312–348, 1899.
2. E. Carvallo, Théorie du mouvement du monocycle et de la bicyclette. Paris,
   France: Gauthier- Villars, 1899.
3. J.P Meijaard, Jim M Papadopoulos, Andy Ruina and A.L Schwab. 2007.
   Linearized dynamics equations for the balance and steer of a bicycle: a
   benchmark and review. Proc. R. Soc. A.4631955–1982
   http://doi.org/10.1098/rspa.2007.1857
4. J. K. Moore, "Human Control of a Bicycle," Doctor of Philosophy, University
   of California, Davis, CA, 2012. http://moorepants.github.io/dissertation
5. J. K. Moore, M. Hubbard, D. L. Peterson, A. L. Schwab, and J. D. G.
   Kooijman, "An Accurate Method of Measuring and Comparing a Bicycle’s
   Physical Parameters," in Proceedings of Bicycle and Motorcycle Dynamics:
   Symposium on the Dynamics and Control of Single Track Vehicles, Delft,
   Netherlands, Oct.  2010.
6. Kooijman, J. D. G., Schwab, A. L. & Meijaard, J. P. 2007 Experimental
   validation of a model of an uncontrolled bicycle.
