"""
===================
Leaning Rider Model
===================

The Carvallo-Whipple bicycle model assumes that the rider is rigidly fixed to
the rear frame of the bicycle. Here the model is extended to include a rigid
body representing the rider's upper body (torso, arms, head) which can lean
relative to the rear frame through a single additional degree of freedom. The
model is defined in [Moore2012]_ chapter "Extensions of the Whipple Model",
section `Leaning rider extension
<https://moorepants.github.io/dissertation/extensions.html#leaning-rider-extension>`_.

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
#
# These parameters include the inertia of the rider's lower body rigidly
# affixed to the bicycle's rear frame :math:`C` and the inertia of the rider's
# upper body is associated with the inverted compound pendulum :math:`G` that
# leans relative to the rear frame of the bicycle :math:`C` through angle
# :math:`q_9` and driven by torque :math:`T_9`.
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
# The state space model is not reduced to the minimal dynamical set, i.e.
# ignorable coordinates are present, so there are zero eigenvalues associated
# with the rows :math:`q_1,q_2,q_3,q_6,q_8,u_6` being present.
evals, evecs = model.calc_eigen()
evals

# %%
# The eigenvalue parts plotted against speed :math:`v` show that the model is
# always unstable due to the inverted pendulum eigenmode of the upper body. But
# the classic weave, capsize, and caster modes are visible and look similar to
# the rigid rider model.
vs = np.linspace(0.0, 10.0, num=401)
ax = model.plot_eigenvalue_parts(hide_zeros=True, sort_modes=False, v=vs)

# %%
# If the stiffness of the rider lean joint :math:`k_9` is increased, the
# dynamics should approach that of the rigid rider model and the self-stability
# can be brought back. Zoom in to see the where the model becomes stable.
k9s = np.linspace(0.0, 300.0, num=301)
ax = model.plot_eigenvalue_parts(hide_zeros=True, sort_modes=False, v=5.5,
                                 k9=k9s, c9=50.0)
ax.set_ylim((-5.0, 5.0))

# %%
# Selecting values for the passive stiffness and damping at the rider upper
# body joint gives dynamics with a small stable speed range.
ax = model.plot_eigenvalue_parts(hide_zeros=True, sort_modes=False, v=vs,
                                 k9=128.0, c9=50.0)

# %%
# Modes of Motion
# ===============
#
# When populated with realistic parameter values the linear Carvallo-Whipple
# model exhibits four characteristic eigenmodes that transition into three
# eigenmodes when two roots bifurcate into an imaginary pair. The rider lean
# degree of freedom adds two real modes at low speed and an additional
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
x0 = np.zeros(12)
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
# The following plot shows the total motion with an initially perturbed roll
# rate at 5.5 m/s.
x0 = np.zeros(12)
x0[8] = 0.5  # u4
ax = model.plot_simulation(times, x0, v=5.5, k9=128.0, c9=50.0)

# %%
# A constant steer torque puts the model into a turn, showing the step
# response.
times = np.linspace(0.0, 5.0, num=101)
x0 = np.zeros(12)
res, inputs = model.simulate(times, x0,
                             input_func=lambda t, x: [0.0, 0.2, 0.0],
                             v=5.5, k9=128.0, c9=50.0)
fig, ax = plt.subplots()
ax.plot(res[:, 0], res[:, 1])
ax.set_xlabel('$q_1$ [m]')
ax.set_ylabel('$q_2$ [m]')
ax.set_aspect('equal')
ax.grid()

# %%
ax = model.plot_simulation(times, x0,
                           input_func=lambda t, x: [0.0, 0.2, 0.0],
                           v=5.5, k9=128.0, c9=50.0)
