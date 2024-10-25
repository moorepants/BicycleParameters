"""
Benchmark
=========

Constructed parameter set from [Meijaard2007]_ that includes the mass and
inertia of a rider.

"""
from pprint import pprint

import numpy as np

from bicycleparameters import Bicycle
from bicycleparameters.io import remove_uncertainties
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from bicycleparameters.models import Meijaard2007Model

data_dir = "../data"

# %%
bicycle = Bicycle("Benchmark", pathToData=data_dir)


# %%
# :math:`v`  seems you have to have some rst math to trigger mathjax inclusion
par = remove_uncertainties(bicycle.parameters['Benchmark'])
par['v'] = 4.6
par_set = Meijaard2007ParameterSet(par, True)
par_set

# %%
ax = par_set.plot_all()

# %%
model = Meijaard2007Model(par_set)

# %%
M, C1, K0, K2 = model.form_reduced_canonical_matrices()
M

# %%
C1

# %%
K0

# %%
K2

# %%
v = np.linspace(0.0, 10.0, num=401)
ax = model.plot_eigenvalue_parts(v=v)
ax.set_ylim((-10.0, 10.0))

# %%
times = np.linspace(0.0, 5.0)
x0 = [0.0, 0.0, 0.5, 0.0]
ax = model.plot_simulation(times, x0)
