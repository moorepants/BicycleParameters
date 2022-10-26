import os

import numpy as np

from ..bicycle import sort_eigenmodes

from dtk import bicycle as dtkbicycle

# local dependencies
import bicycleparameters
from bicycleparameters import bicycle


def test_benchmark_to_canonical():
    M, C1, K0, K2 = dtkbicycle.benchmark_matrices()
    par = dtkbicycle.benchmark_parameters()

    bpM, bpC1, bpK0, bpK2 = bicycle.benchmark_par_to_canonical(par)

    np.testing.assert_allclose(M, bpM)
    np.testing.assert_allclose(C1, bpC1)
    np.testing.assert_allclose(K0, bpK0)
    np.testing.assert_allclose(K2, bpK2)


def test_benchmark_eigenvalues():

    expected = dtkbicycle.benchmark_matrices()
    path_to_package = os.path.split(bicycleparameters.__file__)[0]
    path_to_data = os.path.join(path_to_package, '..', 'data')

    benchmark = bicycleparameters.Bicycle('Benchmark', path_to_data, True,
                                          True)
    M, C1, K0, K2 = benchmark.canonical(nominal=True)

    np.testing.assert_allclose(M, expected[0])
    np.testing.assert_allclose(C1, expected[1])
    np.testing.assert_allclose(K0, expected[2])
    np.testing.assert_allclose(K2, expected[3])


def test_sort_eigenmodes():


    w = np.array([
        [-6.88390903+0.j, -2.92048758+0.j, 3.32151181+0.73198362j, 3.32151181-0.73198362j],
        [ 3.25058404+0.83425573j, 3.25058404-0.83425573j, -7.02339158+0.j, -2.92654704+0.j, ],
        [-7.16278369+0.j, 3.17964562+0.92400675j, 3.17964562-0.92400675j, -2.93267564+0.j]
    ])

    v = np.array([
        [[ 0.00196065+0.j, -0.29800463+0.j,  0.02930766-0.08781924j,  0.02930766+0.08781924j],
         [ 0.14374404+0.j,  0.127017  +0.j, -0.26020427+0.05734294j, -0.26020427-0.05734294j],
         [-0.01349694+0.j,  0.87031883+0.j,  0.16162797-0.27023993j,  0.16162797+0.27023993j],
         [-0.98952091+0.j, -0.37095156+0.j, -0.90624564+0.j,         -0.90624564-0.j]],
        [[ -0.01543099+0.09712139j, -0.01543099-0.09712139j,  0.00191713+0.j, -0.30037796+0.j,],
         [  0.25968865-0.06664856j,  0.25968865+0.06664856j,  0.14094669+0.j,  0.11968475+0.j,],
         [ -0.13118381+0.30282786j, -0.13118381-0.30282786j, -0.01346473+0.j,  0.87907022+0.j,],
         [  0.89974174+0.j,          0.89974174-0.j        , -0.98992376+0.j, -0.35026304+0.j,]],
        [[ 0.00184873+0.j,  0.00157195-0.10415391j,  0.00157195+0.10415391j, -0.30250698+0.j],
         [ 0.13825716+0.j, -0.25897887+0.0752594j , -0.25897887-0.0752594j ,  0.11247147+0.j],
         [-0.01324205+0.j,  0.10123715-0.32972004j,  0.10123715+0.32972004j,  0.88715484+0.j],
         [-0.99030611+0.j, -0.89300123+0.j,         -0.89300123-0.j        , -0.32984235+0.j]]
    ])

    w_expected = np.array([
        [-6.88390903+0.j, -2.92048758+0.j, 3.32151181+0.73198362j, 3.32151181-0.73198362j],
        [-7.02339158+0.j, -2.92654704+0.j, 3.25058404+0.83425573j, 3.25058404-0.83425573j],
        [-7.16278369+0.j, -2.93267564+0.j, 3.17964562+0.92400675j, 3.17964562-0.92400675j]
    ])

    v_expected = np.array([
        [[ 0.00196065+0.j, -0.29800463+0.j,  0.02930766-0.08781924j,  0.02930766+0.08781924j],
         [ 0.14374404+0.j,  0.127017  +0.j, -0.26020427+0.05734294j, -0.26020427-0.05734294j],
         [-0.01349694+0.j,  0.87031883+0.j,  0.16162797-0.27023993j,  0.16162797+0.27023993j],
         [-0.98952091+0.j, -0.37095156+0.j, -0.90624564+0.j,         -0.90624564-0.j]],
        [[ 0.00191713+0.j, -0.30037796+0.j, -0.01543099+0.09712139j, -0.01543099-0.09712139j],
         [ 0.14094669+0.j,  0.11968475+0.j,  0.25968865-0.06664856j,  0.25968865+0.06664856j],
         [-0.01346473+0.j,  0.87907022+0.j, -0.13118381+0.30282786j, -0.13118381-0.30282786j],
         [-0.98992376+0.j, -0.35026304+0.j,  0.89974174+0.j,          0.89974174-0.j]],
        [[ 0.00184873+0.j, -0.30250698+0.j,  0.00157195-0.10415391j,  0.00157195+0.10415391j],
         [ 0.13825716+0.j,  0.11247147+0.j, -0.25897887+0.0752594j , -0.25897887-0.0752594j ],
         [-0.01324205+0.j,  0.88715484+0.j,  0.10123715-0.32972004j,  0.10123715+0.32972004j],
         [-0.99030611+0.j, -0.32984235+0.j, -0.89300123+0.j,         -0.89300123-0.j]]
    ])

    w_sorted, v_sorted = sort_eigenmodes(w, v)

    np.testing.assert_allclose(w_sorted, w_expected)
    np.testing.assert_allclose(v_sorted, v_expected)
