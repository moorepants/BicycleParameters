import numpy.testing as nptest
from dtk import bicycle as dtkbicycle

# local dependencies
from bicycleparameters import bicycle

def test_benchmark_to_canonical():
    M, C1, K0, K2 = dtkbicycle.benchmark_matrices()
    par = dtkbicycle.benchmark_parameters()

    bpM, bpC1, bpK0, bpK2 = bicycle.benchmark_par_to_canonical(par)

    nptest.assert_allclose(M, bpM)
    nptest.assert_allclose(C1, bpC1)
    nptest.assert_allclose(K0, bpK0)
    nptest.assert_allclose(K2, bpK2)
