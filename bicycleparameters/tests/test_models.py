import numpy as np
from pytest import raises

from ..parameter_sets import (Meijaard2007ParameterSet,
                              Meijaard2007WithFeedbackParameterSet)
from ..models import Meijaard2007Model, Meijaard2007WithFeedbackModel

meijaard2007_parameters = {  # dictionary of the parameters in Meijaard 2007
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


def test_parse_parameter_overrides():

    parameter_set = Meijaard2007ParameterSet(meijaard2007_parameters, True)

    model = Meijaard2007Model(parameter_set)

    par, array_keys, array_len = model._parse_parameter_overrides(zH=4.0)
    assert par['zH'] == 4.0
    assert array_keys == []
    assert array_len is None

    par, array_keys, array_len = model._parse_parameter_overrides(
        zH=4.0, IBxx=6.0)
    assert par['zH'] == 4.0
    assert par['IBxx'] == 6.0
    assert array_keys == []
    assert array_len is None

    par, array_keys, array_len = model._parse_parameter_overrides(
        zH=4.0, IBxx=[1.0, 2.0, 3.0])
    assert par['zH'] == 4.0
    assert array_keys == ['IBxx']
    np.testing.assert_allclose(par['IBxx'], [1.0, 2.0, 3.0])
    assert array_len == 3

    par, array_keys, array_len = model._parse_parameter_overrides(
        v=[1.0, 2.0, 3.0])
    np.testing.assert_allclose(par['v'], [1.0, 2.0, 3.0])
    assert array_keys == ['v']
    assert array_len == 3

    par, array_keys, array_len = model._parse_parameter_overrides(
        g=[1.0, 2.0, 3.0])
    np.testing.assert_allclose(par['g'], [1.0, 2.0, 3.0])
    assert array_keys == ['g']
    assert array_len == 3

    par, array_keys, array_len = model._parse_parameter_overrides(
        zH=4.0, v=[1.0, 2.0, 3.0])
    assert par['zH'] == 4.0
    assert array_keys == ['v']
    np.testing.assert_allclose(par['v'], [1.0, 2.0, 3.0])
    assert array_len == 3

    par, array_keys, array_len = model._parse_parameter_overrides(
        zH=[5.0, 6.0, 7.0], v=[1.0, 2.0, 3.0])
    assert array_keys == ['zH', 'v']
    np.testing.assert_allclose(par['v'], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(par['zH'], [5.0, 6.0, 7.0])
    assert array_len == 3

    with raises(ValueError):
        par, array_keys, array_len = model._parse_parameter_overrides(
            zH=[5.0, 6.0], v=[1.0, 2.0, 3.0])


def test_Meijaard2007Model(show=False):

    parameter_set = Meijaard2007ParameterSet(meijaard2007_parameters, True)

    model = Meijaard2007Model(parameter_set)

    M, C1, K0, K2 = model.form_reduced_canonical_matrices()
    assert M.shape == (2, 2)
    assert C1.shape == (2, 2)
    assert K0.shape == (2, 2)
    assert K2.shape == (2, 2)
    M, C1, K0, K2 = model.form_reduced_canonical_matrices(
        w=np.linspace(0.5, 1.5, num=5))
    assert M.shape == (5, 2, 2)
    assert C1.shape == (5, 2, 2)
    assert K0.shape == (5, 2, 2)
    assert K2.shape == (5, 2, 2)
    M, C1, K0, K2 = model.form_reduced_canonical_matrices(
        w=np.linspace(0.5, 1.5, num=10), v=np.linspace(1, 3, num=10))
    assert M.shape == (10, 2, 2)
    assert C1.shape == (10, 2, 2)
    assert K0.shape == (10, 2, 2)
    assert K2.shape == (10, 2, 2)

    A, B = model.form_state_space_matrices()
    assert A.shape == (4, 4)
    assert B.shape == (4, 2)
    A, B = model.form_state_space_matrices(w=np.linspace(0.5, 1.5, num=5))
    assert A.shape == (5, 4, 4)
    assert B.shape == (5, 4, 2)
    A, B = model.form_state_space_matrices(v=np.linspace(0, 10, num=10))
    assert A.shape == (10, 4, 4)
    assert B.shape == (10, 4, 2)

    A, B = model.form_state_space_matrices(v=np.linspace(0, 10, num=10),
                                           g=np.linspace(0, 10, num=10))
    assert A.shape == (10, 4, 4)
    assert B.shape == (10, 4, 2)

    evals, evecs = model.calc_eigen()
    assert evals.shape == (4,)
    assert evecs.shape == (4, 4)
    evals, evecs = model.calc_eigen(g=6.0)
    assert evals.shape == (4,)
    assert evecs.shape == (4, 4)
    evals, evecs = model.calc_eigen(v=np.linspace(0, 10, num=10))
    assert evals.shape == (10, 4)
    assert evecs.shape == (10, 4, 4)
    model.plot_eigenvalue_parts(v=np.linspace(0, 10, num=10))
    model.plot_eigenvectors(v=np.linspace(0, 10, num=4))
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def test_Meijaard2007WithFeedbackModel(show=False):

    parameter_set = Meijaard2007ParameterSet(meijaard2007_parameters, True)

    model = Meijaard2007WithFeedbackModel(parameter_set)
    assert type(model.parameter_set) == Meijaard2007WithFeedbackParameterSet

    M, C1, K0, K2 = model.form_reduced_canonical_matrices()
    assert M.shape == (2, 2)
    assert C1.shape == (2, 2)
    assert K0.shape == (2, 2)
    assert K2.shape == (2, 2)

    A, B = model.form_state_space_matrices()
    assert A.shape == (4, 4)
    assert B.shape == (4, 2)
    A, B = model.form_state_space_matrices(w=np.linspace(0.5, 1.5, num=5))
    assert A.shape == (5, 4, 4)
    assert B.shape == (5, 4, 2)
    A, B = model.form_state_space_matrices(v=np.linspace(0, 10, num=10))
    assert A.shape == (10, 4, 4)
    assert B.shape == (10, 4, 2)

    A, B = model.form_state_space_matrices(v=np.linspace(0, 10, num=10),
                                           g=np.linspace(0, 10, num=10))
    assert A.shape == (10, 4, 4)
    assert B.shape == (10, 4, 2)

    with raises(ValueError):
        # one parameter must be an array
        model.plot_gains()

    model.plot_gains(g=np.linspace(0, 10, num=10))

    model.plot_gains(g=np.linspace(0, 10, num=10),
                     kTdel_phi=np.linspace(0, 10, num=10))

    model.plot_gains(g=np.linspace(0, 10, num=10),
                     v=np.linspace(0, 10, num=10),
                     kTdel_phi=np.linspace(0, 10, num=10))

    if show:
        import matplotlib.pyplot as plt
        plt.show()
