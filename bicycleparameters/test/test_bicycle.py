#!/usr/bin/env python

# standard lib
import os

# external dependencies
import numpy.testing as nptest
from dtk import bicycle as dtkbicycle

# local dependencies
import bicycleparameters
from bicycleparameters import bicycle

def test_benchmark_to_canonical():
    M, C1, K0, K2 = dtkbicycle.benchmark_matrices()
    par = dtkbicycle.benchmark_parameters()

    bpM, bpC1, bpK0, bpK2 = bicycle.benchmark_par_to_canonical(par)

    nptest.assert_allclose(M, bpM)
    nptest.assert_allclose(C1, bpC1)
    nptest.assert_allclose(K0, bpK0)
    nptest.assert_allclose(K2, bpK2)

def test_benchmark_eigenvalues():

    expected = dtkbicycle.benchmark_matrices()
    path_to_package = os.path.split(bicycleparameters.__file__)[0]
    path_to_data = os.path.join(path_to_package, '..', 'data')

    benchmark = bicycleparameters.Bicycle('Benchmark', path_to_data, True,
                                          True)
    M, C1, K0, K2 = benchmark.canonical(nominal=True)

    nptest.assert_allclose(M, expected[0])
    nptest.assert_allclose(C1, expected[1])
    nptest.assert_allclose(K0, expected[2])
    nptest.assert_allclose(K2, expected[3])
