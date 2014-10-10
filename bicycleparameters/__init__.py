# import all the important classes and functions from the bicycleparametes
# module
from .main import Bicycle
from .plot import plot_eigenvalues, compare_bode_bicycles
from .tables import Table
from .version import __version_info__, __version__

# the modules that are imported when 'from bicycleparameters import *'
__all__ = ['main',
           'geometry',
           'io',
           'period',
           'rider',
           'bicycle',
           'com',
           'inertia',
           'plot',
           'tables']
