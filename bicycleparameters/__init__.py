# import all the important classes and functions from the bicycleparametes module
from main import Bicycle
from plot import plot_eigenvalues

# the modules that are imported when 'from bicycleparameters import *'
__all__ = ['main',
           'geometry',
           'io',
           'period',
           'rider',
           'bicycle',
           'com',
           'inertia',
           'plot']

# specify the version for the package
__version_info__ = (0, 1, 3)
__version__ = '.'.join(map(str, __version_info__))
