import numpy as np
import matplotlib.pyplot as plt

from bicycleparameters.bicycle import benchmark_par_to_canonical
from bicycleparameters.models import Meijaard2007Model
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet


class SteerAssistModel(Meijaard2007Model):
    """

    Tdel_total = -kphidot*phidot - kphi*phi + Tdel

    """

    def form_reduced_canonical_matrices(self, **parameter_overrides):

        par, array_key, array_val = self._parse_parameter_overrides(
            **parameter_overrides)

        print(array_key)

        if array_val is not None:
            M = np.zeros((len(array_val), 2, 2))
            C1 = np.zeros((len(array_val), 2, 2))
            K0 = np.zeros((len(array_val), 2, 2))
            K2 = np.zeros((len(array_val), 2, 2))
            for i, val in enumerate(array_val):
                par[array_key] = val
                M[i], C1[i], K0[i], K2[i] = benchmark_par_to_canonical(par)
                if array_key == 'kphidot':
                    C1[i, 1, 0] = C1[i, 1, 0] + val
                elif array_key == 'kphi':
                    K0[i, 1, 0] = K0[i, 1, 0] + val
                else:
                    C1[i, 1, 0] = C1[i, 1, 0] + par['kphidot']
                    K0[i, 1, 0] = K0[i, 1, 0] + par['kphi']
            return M, C1, K0, K2
        else:
            M, C1, K0, K2 = benchmark_par_to_canonical(par)
            C1[1, 0] = C1[1, 0] + par['kphidot']
            K0[1, 0] = K0[1, 0] + par['kphi']
            return M, C1, K0, K2


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
    'v': 2.0,
    'w': 1.02,
    'xB': 0.3,
    'xH': 0.9,
    'zB': -0.9,
    'zH': -0.7,
    'kphi': 0.0,
    'kphidot': 0.0,
}


parameter_set = Meijaard2007ParameterSet(meijaard2007_parameters, True)

model = SteerAssistModel(parameter_set)
model.plot_eigenvalue_parts(kphidot=np.linspace(-20.0, 10.0, num=1000))
model.plot_eigenvalue_parts(v=np.linspace(0.0, 10.0, num=1000))
model.plot_eigenvalue_parts(kphidot=-10.0, v=np.linspace(0.0, 10.0, num=1000))

plt.show()
