Release Notes
=============

1.1.0
-----

- Introduced a Dash based web application for interactive use.

  - https://github.com/moorepants/BicycleParameters/pull/34
  - https://github.com/moorepants/BicycleParameters/pull/88

- Dropped support for Python 2.7, 3.4, 3.5, 3.6, and 3.7. Added support for
  Python 3.8, 3.9, 3.10, 3.11, 3.12.
- Bumped dependency minimum verions to match Ubuntu 22.04.
- Moved to mamba and Github action based continuous integration testing.
- Introduced new parameter_sets and models modules for future class hierarchies
  and better design.
- ``bicycleparameters/test/`` moved to ``bicycleparameters/tests/``.
- Removed remaining NumPy ``matrix()`` calls.
- Improved some of the matplotlib plots.

1.0.0
-----

- Support Python 3

0.2.0
-----

- Commands using the state space form of the Whipple model have been reordered
  to [roll angle, steer angle, roll rate, steer rate]
- Added another rider's measurments.
- Added a module for printing tables of data.
- Added the Gyrobike and the ability to manage it's flywheel rigidbody.
- Fixed a bug in `calculate_abc_geometry()` that gave incorrect geometry
  values.
- Handles two additional points for the Davis Instrumented Bicycle.
- Added a child sized person based on scaling Charlie's measurements.
- Added Bode plot commands.
- Added nominal output options for several methods.
- Added a dependency to DynamicistToolKit
- Updated core dependencies to a minimum from the Ubuntu 12.04 release.
- Tested with DTK 0.1.0 to 0.3.5.
- Added Travis support.
- The minimum yeadon version is bumped to 1.1.1 and code updated to reflect the
  new yeadon api.
- The minimum version of uncertainties is bumped to 2.0.

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

- Initial release.
