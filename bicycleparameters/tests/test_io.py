import os
import numpy as np
from bicycleparameters.io import load_pendulum_mat_file

TESTS_DIR = os.path.dirname(__file__)


def test_load_pendulum_mat_file():
    path = os.path.join(TESTS_DIR, 'sample_data',
                        'BrowserForkCompoundFirst1.mat')

    d = load_pendulum_mat_file(path)

    assert d['angle'] == 'First'
    assert d['ActualRate'] == 1000
    assert d['bicycle'] == 'BrowserIns'
    assert d['data'].shape == (30000,)
    np.testing.assert_allclose(
        d['data'][0:5],
        np.array([0.33034183, 0.32525861, 0.31509219, 0.31509219, 0.33034183]))
    assert d['duration'] == 30
    assert d['filename'] == 'BrowserInsForkCompoundFirst1'
    assert d['notes'] == ''
    assert d['part'] == 'Fork'
    assert d['pendulum'] == 'Compound'
    assert d['trial'] == '1'
