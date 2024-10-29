"""
================================
Benchmark Carvallo-Whipple Model
================================

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
par = remove_uncertainties(bicycle.parameters['Benchmark'])
par['v'] = 4.6
par_set = Meijaard2007ParameterSet(par, True)
par_set

# %%
ax = par_set.plot_all()

# %%
# Linear Carvollo-Whipple Model
# =============================
#
# Create a model that represents linear Carvallo-Whipple model presented in
# [Meijaard2007]_.
model = Meijaard2007Model(par_set)

# %%
# This model can be written in this form:
#
# .. math::
#
#    \mathbf{M}\ddot{\vec{q}} + v\mathbf{C}_1\dot{\vec{q}} + \left[ g \mathbf{K}_0 + v^2 \mathbf{K}_2 \right] \vec{q} = 0
#
# where the essential coordinates are the roll angle and steer angle,
# respectively:
#
# .. math::
#
#    \bar{q} = [\phi \quad \delta]^T
#
# The model can calculate the cofficient matrices, which should match the
# paper's results when using the the parameter values from the paper:
M, C1, K0, K2 = model.form_reduced_canonical_matrices()
M

# %%
C1

# %%
K0

# %%
K2

# %%
# The root locus can be plotted as a function of any model parameter. The root
# locus plot showing the real and imaginary parts of each eigenvalues as a
# function of the parameter speed reproduces the plot in [Meijaard2007]_. This
# plot is most often used to show how stability is depedendent on speed.
v = np.linspace(0.0, 10.0, num=401)
ax = model.plot_eigenvalue_parts(v=v, colors=['C0', 'C0', 'C1', 'C2'],
                                 hide_zeros=True)
ax.axvline(0.0, color='k')
ax.axvline(2.0, color='k')
ax.axvline(5.0, color='k')
ax.axvline(8.0, color='k')
ax.set_ylim((-10.0, 10.0))

# %%
# Modes of Motion
# ===============
#
# Low Speed, Unstable Inverted Pendlum
# ------------------------------------
# Before the bifurcation point, there are four real eigenvalues.
ax = model.plot_eigenvectors(v=0.0)
# %%
times = np.linspace(0.0, 1.0)
ax = model.plot_mode_simulations(times, v=0.0)

# %%
ax = model.plot_simulation(times, [0.01, 0.0, 0.0, 0.0], v=0.0)

# %%
# Low Speed, Unstable Weave
# -------------------------
ax = model.plot_eigenvectors(v=2.0)
# %%
times = np.linspace(0.0, 1.0)
ax = model.plot_mode_simulations(times, v=2.0)

# %%
# Stable
# ------
ax = model.plot_eigenvectors(v=5.0)
# %%
times = np.linspace(0.0, 5.0)
ax = model.plot_mode_simulations(times, v=5.0)

# %%
# High Speed, Unstable Capsize
# ----------------------------
ax = model.plot_eigenvectors(v=8.0)
# %%
times = np.linspace(0.0, 5.0)
ax = model.plot_mode_simulations(times, v=8.0)

# %%
x0 = [0.0, 0.0, 0.5, 0.0]
ax = model.plot_simulation(times, x0)
