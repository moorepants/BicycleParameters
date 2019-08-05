import numpy as np

from ..parameter_sets import BenchmarkParameterSet

benchmark_parameters = {  # dictionary of the parameters in Meijaard 2007
    'IBxx': 9.2,
    'IBxz': 2.4,
    'IByy': 11.0,
    'IBzz': 2.8,
    'IFxx': 0.1405,
    'IFyy': 0.28,
    'IHxx': 0.05892,
    'IHxz': -0.00756,
    'IHyy': 0.06,
    'IHzz': 0.00708,
    'IRxx': 0.0603,
    'IRyy': 0.12,
    'c': 0.08,
    'g': 9.81,
    'lam': np.pi/10.0,
    'mB': 85.0,
    'mF': 3.0,
    'mH': 4.0,
    'mR': 2.0,
    'rF': 0.35,
    'rR': 0.3,
    'v': 5.0,
    'w': 1.02,
    'xB': 0.3,
    'xH': 0.9,
    'zB': -0.9,
    'zH': -0.7,
}

# bike only
browser_par = {
    'IBxx': 0.52962890621,
    'IBxz': -0.116285607878,
    'IByy': 1.3163960125,
    'IBzz': 0.756786895402,
    'IFxx': 0.0883826870796,
    'IFyy': 0.149221207336,
    'IHxx': 0.25335539588,
    'IHxz': -0.0720217263293,
    'IHyy': 0.245827908036,
    'IHzz': 0.0955686343473,
    'IRxx': 0.0904114316323,
    'IRyy': 0.152391250767,
    'c': 0.0685808540382,
    'g': 9.81,
    'lam': 0.399680398707,
    'mB': 9.86,
    'mF': 2.02,
    'mH': 3.22,
    'mR': 3.11,
    'rF': 0.34352982332,
    'rR': 0.340958858855,
    'w': 1.121,
    'xB': 0.275951285677,
    'xH': 0.866949640247,
    'zB': -0.537842424305,
    'zH': -0.748236400835,
}

# bike only
pista_par = {
    'IBxx': 0.289968529291,
    'IBxz': 0.0502317583637,
    'IByy': 0.475817094274,
    'IBzz': 0.249418014295,
    'IFxx': 0.0553476418952,
    'IFyy': 0.106185287785,
    'IHxx': 0.0979884309288,
    'IHxz': -0.00440828467861,
    'IHyy': 0.0691907586693,
    'IHzz': 0.0396107377431,
    'IRxx': 0.0552268228143,
    'IRyy': 0.0764034890242,
    'c': 0.0617300113263,
    'g': 9.81,
    'lam': 0.275762021815,
    'mB': 4.49,
    'mF': 1.58,
    'mH': 2.27,
    'mR': 1.38,
    'rF': 0.333838861345,
    'rR': 0.332088156971,
    'w': 0.989,
    'xB': 0.382963782674,
    'xH': 0.906321313298,
    'zB': -0.476798507146,
    'zH': -0.732376356772,
}


def test_BenchmarkParameterSet(plot=False):
    parameter_set = BenchmarkParameterSet(benchmark_parameters, True)
    assert parameter_set.includes_rider == True
    assert parameter_set.parameters['v'] == 5.0

    parameter_set = BenchmarkParameterSet(browser_par, False)
    ax = parameter_set.plot_geometry()
    ax = parameter_set.plot_principal_radii_of_gyration(ax=ax)
    ax = parameter_set.plot_mass_centers(ax=ax)
    ax = parameter_set.plot_inertia_ellipsoids(ax=ax)

    if plot:
        import matplotlib.pyplot as plt
        plt.show()
