"""
===================
Leaning Rider Model
===================

The Carvallo-Whipple bicycle model assumes that the rider is rigidly fixed to
the rear frame of the bicycle. Here the model is extended to include a rigid
body representing the rider's upper body (torso, arms, head) which can lean
relative to the rear frame through a single additional degree of freedom.

"""
import numpy as np
import matplotlib.pyplot as plt

from bicycleparameters.parameter_dicts import mooreriderlean2012_browser_jason
from bicycleparameters.parameter_sets import MooreRiderLean2012ParameterSet
from bicycleparameters.models import MooreRiderLean2012Model

np.set_printoptions(precision=3, suppress=True)

# %%
# Load Parameters
# ===============
par_set = MooreRiderLean2012ParameterSet(mooreriderlean2012_browser_jason)
par_set

# %%
# Construct Model and Inspect State Space
# =======================================
model = MooreRiderLean2012Model(par_set)
A, B = model.form_state_space_matrices()

# %%
model.state_vars

# %%
A

# %%
model.input_vars

# %%
B

# %%
# Eigenvalues and Eigenvectors
# ============================
evals, evecs = model.calc_eigen()
evals

# %%
# The eigenvalue parts plotted against speed :math:`v` show that the model is
# always unstable due to the inverted pendulum eigenmode of the upper body.
vs = np.linspace(0.0, 10.0, num=401)
ax = model.plot_eigenvalue_parts(hide_zeros=True, v=vs)

# %%
# If the stiffness of the rider lean joint is increased, the dyamics should
# approach that of the rigid rider model and the self-stability can be brought
# back.

# TODO : The eigsort may be catching on the zero eigenvalues.
# TODO : Stable region does not show.
k9s = np.linspace(0.0, 300.0, num=301)
ax = model.plot_eigenvalue_parts(hide_zeros=True, v=5.5, k9=k9s, c9=50.0)

# %%
# Zoom in to see the where the model becomes stable.
ax = model.plot_eigenvalue_parts(hide_zeros=True, v=5.5, k9=k9s, c9=50.0)
ax.set_ylim((-5.0, 5.0))

# %%
# Selecting values for the passive stiffness and damping at the rider upper
# body joint gives dynamics with a small stable speed range.
ax = model.plot_eigenvalue_parts(hide_zeros=True, v=vs, k9=128.0, c9=50.0)

# %%
# Modes of Motion
# ===============
#
# When populated with realistic parameter values the linear Carvallo-Whipple
# model exhibits four characteristic eigenmodes that transition into three
# eigenmodes when two roots bifurcate into an imaginary pair. The rider lean
# degree of freedom adds to real modes at low speed and an additional
# oscillatory mode at high speed.
#
# Low Speed, Unstable Inverted Pendulum
# -------------------------------------
ax = model.plot_eigenvectors(hide_zeros=True, v=0.0, k9=128.0, c9=50.0)

# %%
times = np.linspace(0.0, 1.0)
ax = model.plot_mode_simulations(times, hide_zeros=True, v=0.0, k9=128.0,
                                 c9=50.0)

# %%
# The total motion shows that the bicycle falls over.
x0 = np.zeros(13)
x0[3] = np.deg2rad(1.0)  # q4
ax = model.plot_simulation(times, x0, v=0.0, k9=128.0, c9=50.0)

# %%
# Low Speed, Unstable Weave
# -------------------------
ax = model.plot_eigenvectors(hide_zeros=True, v=2.0, k9=128.0, c9=50.0)

# %%
times = np.linspace(0.0, 1.0)
ax = model.plot_mode_simulations(times, hide_zeros=True, v=2.0, k9=128.0,
                                 c9=50.0)
# %%
# Stable Eigenmodes
# -----------------
ax = model.plot_eigenvectors(hide_zeros=True, v=5.5, k9=128.0, c9=50.0)

# %%
times = np.linspace(0.0, 5.0, num=501)
ax = model.plot_mode_simulations(times, hide_zeros=True, v=5.5, k9=128.0,
                                 c9=50.0)

# %%
# High Speed, Unstable Capsize
# ----------------------------
ax = model.plot_eigenvectors(hide_zeros=True, v=8.0, k9=128.0, c9=50.0)

# %%
ax = model.plot_mode_simulations(times, hide_zeros=True, v=8.0, k9=128.0,
                                 c9=50.0)
# %%
# Total Motion Simulation
# -----------------------
#
# The following plot shows the total motion with a perturbed initial condition
# at 5.5 m/s.
x0 = np.zeros(13)
x0[9] = 0.5  # u4
x0[10] = -5.5/model.parameter_set.parameters['rr']  # u6
ax = model.plot_simulation(times, x0, v=5.5, k9=128.0, c9=50.0)

# %%
# A constant steer torque puts the model into a turn.
times = np.linspace(0.0, 10.0, num=101)
res, inputs = model.simulate(times, x0,
                             input_func=lambda t, x: [0.0, 0.0, 1.0, 0.0],
                             v=5.5, k9=128.0, c9=50.0)
fig, ax = plt.subplots()
ax.plot(res[:, 0], res[:, 1])
ax.grid()

# %%
ax = model.plot_simulation(times, x0,
                           input_func=lambda t, x: [0.0, 0.0, 1.0, 0.0],
                           v=5.5, k9=128.0, c9=50.0)
