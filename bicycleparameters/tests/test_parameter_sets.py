import os

import yaml
import numpy as np
import matplotlib.pyplot as plt

from ..parameter_sets import Meijaard2007ParameterSet, Moore2019ParameterSet

PARDIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'parameter_sets')


def test_Meijaard2007ParameterSet(plot=False):

    with open(os.path.join(PARDIR_PATH, 'benchmark-benchmark.yml'), 'r') as f:
        benchmark_par = yaml.load(f, Loader=yaml.FullLoader)['values']

    pset = Meijaard2007ParameterSet(benchmark_par, True)

    assert pset.includes_rider is True
    assert pset.parameters['v'] == 5.0

    expected_IH = np.array([[0.05892, 0.0, -0.00756],
                            [0.0, 0.06, 0.0],
                            [-0.00756, 0.0, 0.00708]])
    np.testing.assert_allclose(pset.form_inertia_tensor('H'),
                               expected_IH)

    expected_comB = np.array([[0.3], [0.0], [-0.9]])
    expected_comF = np.array([[1.02], [0.0], [-0.35]])
    expected_comH = np.array([[0.9], [0.0], [-0.7]])
    expected_comR = np.array([[0.0], [0.0], [-0.3]])

    np.testing.assert_allclose(expected_comB,
                               pset.form_mass_center_vector('B'))
    np.testing.assert_allclose(expected_comF,
                               pset.form_mass_center_vector('F'))
    np.testing.assert_allclose(expected_comH,
                               pset.form_mass_center_vector('H'))
    np.testing.assert_allclose(expected_comR,
                               pset.form_mass_center_vector('R'))

    np.testing.assert_allclose(pset.form_mass_center_vector('H'),
                               pset.mass_center_of('H'))

    com_total = pset.mass_center_of('F', 'R')

    x = (1.02*3.0 + 0.0*2.0)/(3.0 + 2.0)
    y = 0.0
    z = (-0.35*3.0 + -0.3*2.0)/(3.0 + 2.0)

    np.testing.assert_allclose(com_total, np.array([x, y, z]))

    if plot:
        ax = pset.plot_geometry()
        ax = pset.plot_principal_radii_of_gyration(ax=ax)
        ax = pset.plot_principal_inertia_ellipsoids(ax=ax)
        ax = pset.plot_mass_centers(ax=ax)
        plt.show()


def test_Moore2019ParameterSet(plot=False):

    with open(os.path.join(PARDIR_PATH,
                           'principal-browserjason.yml'), 'r') as f:
        principal_par = yaml.load(f, Loader=yaml.FullLoader)['values']

    pset = Moore2019ParameterSet(principal_par)

    ax = pset.plot_geometry()
    ax = pset.plot_person_diamond(ax=ax)
    ax = pset.plot_principal_radii_of_gyration(ax=ax)
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


def test_conversion(plot=False):
    with open(os.path.join(PARDIR_PATH,
                           'principal-browserjason.yml'), 'r') as f:
        par_dict = yaml.load(f, Loader=yaml.FullLoader)['values']
    pset = Moore2019ParameterSet(par_dict)
    pset.plot_all()
    bench_pset = pset.to_benchmark()
    bench_pset.plot_all()
    if plot:
        plt.show()
