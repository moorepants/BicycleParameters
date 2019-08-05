import numpy as np
import pytest

from ..parameter_sets import BenchmarkParameterSet
from ..models import MinimalLinearWhippleCarvalloModel

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


def test_MinimalLinearWhippleCarvalloModel():

    parameter_set = BenchmarkParameterSet(benchmark_parameters, True)

    model = MinimalLinearWhippleCarvalloModel(parameter_set)

    M, C1, K0, K2 = model.form_benchmark_canonical_matrices()
    assert M.shape == (2, 2)
    assert C1.shape == (2, 2)
    assert K0.shape == (2, 2)
    assert K2.shape == (2, 2)
    M, C1, K0, K2 = model.form_benchmark_canonical_matrices(
        w=np.linspace(0.5, 1.5, num=5))
    assert M.shape == (5, 2, 2)
    assert C1.shape == (5, 2, 2)
    assert K0.shape == (5, 2, 2)
    assert K2.shape == (5, 2, 2)
    with pytest.raises(ValueError):
        model.form_benchmark_canonical_matrices(w=np.linspace(0.5, 1.5),
                                                v=np.linspace(1, 3))

    A, B = model.form_state_space_matrices()
    assert A.shape == (4, 4)
    assert B.shape == (4, 2)
    A, B = model.form_state_space_matrices(w=np.linspace(0.5, 1.5, num=5))
    assert A.shape == (5, 4, 4)
    assert B.shape == (5, 4, 2)
    A, B = model.form_state_space_matrices(v=np.linspace(0, 10, num=10))
    assert A.shape == (10, 4, 4)
    assert B.shape == (10, 4, 2)

    evals, evecs = model.eigen()
    assert evals.shape == (4,)
    assert evecs.shape == (4, 4)
    evals, evecs = model.eigen(g=6.0)
    assert evals.shape == (4,)
    assert evecs.shape == (4, 4)
    evals, evecs = model.eigen(v=np.linspace(0, 10, num=10))
    assert evals.shape == (10, 4)
    assert evecs.shape == (10, 4, 4)
    model.plot_eigenvalue_parts(v=np.linspace(0, 10, num=10))
