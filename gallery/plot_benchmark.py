"""
================================
Benchmark Carvallo-Whipple Model
================================

[Meijaard2007]_ presents a benchmark minimal linear Carvallo-Whipple bicycle
model with 27 unique constant parameters and two coordinates: roll angle and
steer angle.

"""
import numpy as np

from bicycleparameters import Bicycle
from bicycleparameters.io import remove_uncertainties
from bicycleparameters.parameter_sets import Meijaard2007ParameterSet
from bicycleparameters.models import Meijaard2007Model

data_dir = "../data"

# %%
# Benchmark Parameter Values
# ==========================
# First, create a :py:class:`Bicycle` object from the parameter file in the
# data directory.
bicycle = Bicycle("Benchmark", pathToData=data_dir)

# %%
# Strip the uncertainties from the parameters and add a speed parameter
# :math:`v`, then create a parameter set from the parameter values.
par = remove_uncertainties(bicycle.parameters['Benchmark'])
par['v'] = 4.6
par_set = Meijaard2007ParameterSet(par, True)
par_set

# %%
# The bicycle geometry and representations of the inertia of the four rigid
# bodies can be visualized with the parameter set plot methods.
ax = par_set.plot_all()

# %%
# Linear Carvollo-Whipple Model
# =============================
#
# Create a model that represents linear Carvallo-Whipple model presented in
# [Meijaard2007]_ and populate it with the parameter values in the parameter
# set..
model = Meijaard2007Model(par_set)

# %%
# This model can be written in this form:
#
# .. math::
#
#    \mathbf{M}\ddot{\vec{q}} +
#    v\mathbf{C}_1\dot{\vec{q}} +
#    \left[ g \mathbf{K}_0 + v^2 \mathbf{K}_2 \right] \vec{q} =
#    \vec{F}
#
# where the essential coordinates are the roll angle :math:`\phi` and steer
# angle :math:`\delta`, respectively:
#
# .. math::
#
#    \vec{q} = [\phi \quad \delta]^T
#
# and the forcing terms are:
#
# .. math::
#
#    \vec{F} = [T_\phi \quad T_\delta]^T
#
# The model can calculate the mass, damping, and stiffness cofficient matrices,
# which should match the paper's results when using the matching parameter
# values:
M, C1, K0, K2 = model.form_reduced_canonical_matrices()
M

# %%
C1

# %%
K0

# %%
K2

# %%
# The root locus can be plotted as a function of any model parameter. This plot
# is most often used to show how stability is dependent on the speed
# parameter.The blue lines indicate the weave eigenmode, the green the capsize
# eigenmode, and the orange the caster eigenmode. Vertical dashed lines
# indicate speeds that are examined further below.
v = np.linspace(0.0, 10.0, num=401)
ax = model.plot_eigenvalue_parts(v=v, colors=['C0', 'C0', 'C1', 'C2'],
                                 hide_zeros=True)
ax.axvline(0.0, color='k', linestyle='--')
ax.axvline(2.0, color='k', linestyle='--')
ax.axvline(5.0, color='k', linestyle='--')
ax.axvline(8.0, color='k', linestyle='--')
ax.set_ylim((-10.0, 10.0))

# %%
# Modes of Motion
# ===============
#
# Low Speed, Unstable Inverted Pendlum
# ------------------------------------
# Before the bifurcation point into the weave mode, there are four real
# eigenvalues.
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
