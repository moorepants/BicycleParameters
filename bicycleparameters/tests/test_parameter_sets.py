import yaml
import numpy as np
import matplotlib.pyplot as plt

from ..parameter_sets import BenchmarkParameterSet, PrincipalParameterSet

with open('parameter_sets/benchmark-benchmark.yml', 'r') as f:
    benchmark_benchmark = yaml.load(f)['values']

with open('parameter_sets/benchmark-browser.yml', 'r') as f:
    browser_par = yaml.load(f)

with open('parameter_sets/benchmark-pista.yml', 'r') as f:
    benchmark_pista = yaml.load(f)

with open('parameter_sets/benchmark-pistarider.yml', 'r') as f:
    benchmark_pistarider = yaml.load(f)

with open('parameter_sets/benchmark-realizedopttwo.yml', 'r') as f:
    benchmark_realizedopttwo = yaml.load(f)

with open('parameter_sets/benchmark-pistarideroptimized3ms.yml', 'r') as f:
    benchmark_pistarideroptimized3ms = yaml.load(f)


def test_BenchmarkParameterSet(plot=False):
    pset = BenchmarkParameterSet(benchmark_benchmark, True)
    assert pset.includes_rider is True
    assert pset.parameters['v'] == 5.0
    expected_tensor = np.array([[0.05892, 0.0, -0.00756],
                                [0.0, 0.06, 0.0],
                                [-0.00756, 0.0, 0.00708]])
    np.testing.assert_allclose(pset.form_inertia_tensor('H'), expected_tensor)
    expected_com = np.array([[0.9], [0.0], [-0.7]])
    np.testing.assert_allclose(pset.form_mass_center_vector('H'), expected_com)
    np.testing.assert_allclose(pset.form_mass_center_vector('H'), pset.mass_center_of('H'))
    pset.mass_center_of('B', 'F', 'H', 'R')

    pause

    pset = BenchmarkParameterSet(browser_par, False)
    pset = BenchmarkParameterSet(benchmark_pista, False)

    pset = BenchmarkParameterSet(extendedoptc_par, True)

    pset = BenchmarkParameterSet(benchmark_pistarider, True)
    pset = BenchmarkParameterSet(benchmark_pistarideroptimized3ms, True)
    pset = BenchmarkParameterSet(benchmark_realizedopttwo, True)

    ax = pset.plot_geometry()
    ax = pset.plot_principal_radii_of_gyration(ax=ax)
    ax = pset.plot_inertia_ellipsoids(ax=ax)
    ax = pset.plot_mass_centers(ax=ax)

    if plot:
        plt.show()

principal_par_jasonbrowser_dict = {
    'alphaD': 1.1722101094171953,
    'alphaH': 0.3699481738478786,
    'alphaP': 0.18617531745505142,
    'c': 0.0685808540382,
    'g': 9.81,

    'kDaa': 0.28586689,
    'kDbb': 0.22079007,
    'kDyy': 0.36538846,

    'kFaa': np.sqrt(0.0883826870796/2.02),
    'kFyy': np.sqrt(0.149221207336/2.02),

    'kHaa': 0.29556015,
    'kHbb': 0.14493343,
    'kHyy': 0.27630431,

    'kPyy': 0.36796654,
    'kPaa': 0.36717241,
    'kPbb': 0.15276369,

    'kRaa': np.sqrt(0.0904114316323/3.11),
    'kRyy': np.sqrt(0.152391250767/3.11),

    'lP': 1.728,
    'lam': 0.399680398707,
    'mD': 9.86,
    'mF': 2.02,
    'mH': 3.22,
    'mP': 83.50000000000001,
    'mR': 3.11,
    'rF': 0.34352982332,
    'rR': 0.340958858855,
    'v': 3.0,
    'w': 1.121,
    'wP': 0.483,
    'xD': 0.275951285677,
    'xH': 0.866949640247,
    'xP': 0.3157679154070924,
    'xR': 0.0,
    'yD': 0.0,
    'yF': 0.0,
    'yH': 0.0,
    'yP': 0.0,
    'yR': 0.0,
    'zD': -0.537842424305,
    'zH': -0.748236400835,
    'zP': -1.0989885659819278,
}

# best of 5 m/s Jason + Browser
principal_par_jasonbrowser_dict = {
'alphaD':  0.91544,
'alphaH': -1.2647,
'alphaP':  1.1497,
'c': -0.0050384,
'g':  9.8100,
'kDaa':  1.2179,
'kDbb':  1.3868,
'kDyy':  1.5585,
'kFyy':  0.25170,
'kFaa':  0.12585,
'kFbb':  0.12585,

'kHaa':  0.049106,
'kHbb':  1.1451,
'kHyy':  0.38324,

'kPaa':  0.36797,
'kPbb':  0.15276,
'kPyy':  0.36717,
'kRyy':  0.50612,
'kRaa':  0.25306,
'kRbb':  0.25306,
'lP':  1.7280,
'lam':  0.27137,
'mD':  11.534,
'mF':  4.4038,
'mH':  0.25000,
'mP':  83.500,
'mR':  8.3601,
'rF':  0.25170,
'rR':  0.50612,
'v':  5.0,
'w':  0.84711,
'wP':  0.48300,
'xD':  0.28749,
'xF':  0.84711,
'xH':  0.53227,
'xP':  0.27642,
'xR': 0,
'yD': 0,
'yF': 0,
'yH': 0,
'yP': 0,
'yR': 0,
'zD': -2.7190,
'zF': -0.25170,
'zH': -0.78065,
'zP': -0.45256,
'zR': -0.50612,
}


with open('parameter_sets/benchmark-extendedoptc.yml', 'r') as f:
    extendedoptc_par = yaml.load(f)

with open('parameter_sets/principal-extendedoptf.yml', 'r') as f:
    principal_extendedoptf = yaml.load(f)


def test_conversion(plot=False):
    with open('parameter_sets/principal-browserjason.yml', 'r') as f:
        par_dict = yaml.load(f)['values']
    pset = PrincipalParameterSet(par_dict)
    pset.plot_all()
    bench_pset = pset.to_benchmark()
    bench_pset.plot_all()
    if plot:
        plt.show()


def test_PrincipalParameterSet(plot=False):

    pset = PrincipalParameterSet(principal_extendedoptf)

    ax = pset.plot_geometry()
    ax = pset.plot_person_diamond(ax=ax)
    #ax = pset.plot_principal_radii_of_gyration(ax=ax)
    ax = pset.plot_body_mass_center('D', ax=ax)
    ax = pset.plot_body_mass_center('F', ax=ax)
    ax = pset.plot_body_mass_center('H', ax=ax)
    ax = pset.plot_body_mass_center('P', ax=ax)
    ax = pset.plot_body_mass_center('R', ax=ax)
    ax = pset.plot_body_principal_inertia_ellipsoid('D', ax=ax)
    ax = pset.plot_body_principal_inertia_ellipsoid('P', ax=ax)
    ax = pset.plot_body_principal_inertia_ellipsoid('H', ax=ax)


    if plot:
        plt.show()
