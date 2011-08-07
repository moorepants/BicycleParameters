# import all the classes and functions from the module
from bicycleparameters import *
# remove the bicycleparameters and inertia modules because we already imported
# all the stuff from it
del bicycleparameters, inertia

# specify the version for the package
__version_info__ = (0, 1, 0)
__version__ = '.'.join(map(str, __version_info__))
