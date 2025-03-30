r"""
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

# %%
# Load Parameters
# ===============
par_set = MooreRiderLean2012ParameterSet(mooreriderlean2012_browser_jason)
par_set

# %%
# Construct Model
# ===============
model = MooreRiderLean2012Model(par_set)

# %%
# State Space
# ===========
A, B = model.form_state_space_matrices(v=5.5)
A

# %%
B

# %%
# Eigenvalues and Eigenvectors
# ============================
evals, evecs = model.calc_eigen(v=5.5)
evals

# %%
# The eigenvalue parts plotted against speed show that the model is always
# unstable due to the inverted pendulum eigenmode of the upper body.
vs = np.linspace(0.0, 10.0, num=401)
ax = model.plot_eigenvalue_parts(v=vs)

# %%
# If the stiffness of the rider lean joint is increased, the dyamics should
# approach that of the rigid rider model and the self-stability can be brought
# back.
k9s = np.linspace(0.0, 300.0, num=301)
ax = model.plot_eigenvalue_parts(hide_zeros=True, v=5.5, k9=k9s, c9=50.0)

# %%
# Selecting values for the passive stiffness and damping at the rider upper
# body joint gives dynamics with a small stable speed range.
ax = model.plot_eigenvalue_parts(hide_zeros=True, v=vs, k9=128.0, c9=50.0,)

# %%
# Modes of Motion
# ===============
#
# When populated with realistic parameter values the linear Carvallo-Whipple
# model exhibits four characteristic eigenmodes that transition into three
# eigenmodes when two roots bifurcate into an imaginary pair.
#
# Low Speed, Unstable Inverted Pendulum
# -------------------------------------
#
# Before the bifurcation point into the weave mode, there are four real
# eigenvalues that describe the motion.
ax = model.plot_eigenvectors(hide_zeros=True, v=0.0, k9=128.0, c9=50.0)

# %%
# The unstable eigenmodes dominate the motion and this results in the steer
# angle growing rapidly and the roll angle following suite, but slower.
times = np.linspace(0.0, 1.0)
ax = model.plot_mode_simulations(times, hide_zeros=True, v=0.0, k9=128.0,
                                 c9=50.0)

# %%
# The total motion shows that the bicycle falls over, as expected.
x0 = np.zeros(13)
x0[3] = np.deg2rad(1.0)  # q4
ax = model.plot_simulation(times, x0, v=0.0, k9=128.0, c9=50.0)

# %%
# Low Speed, Unstable Weave
# -------------------------
#
# Moving up in speed past the bifurcation, the bicycle has an unstable
# oscillatory eigenmode that is called "weave". The steer angle has about twice
# the magnitude as the roll angle and they both grow together with close to 90
# degrees phase. The other two stable real valued eigenmodes are called
# "capsize" and "caster", with capsize being primarily a roll motion and caster
# a steer motion.
ax = model.plot_eigenvectors(hide_zeros=True, v=2.0, k9=128.0, c9=50.0)
# %%
times = np.linspace(0.0, 1.0)
ax = model.plot_mode_simulations(times, hide_zeros=True, v=2.0, k9=128.0,
                                 c9=50.0)

# %%
# Stable Eigenmodes
# -----------------
#
# Increasing speed further shows that the weave eigenmode becomes stable,
# making all eigenmodes stable, and thus the total motion is stable. The speed
# regime where this is true is called the "self-stable" speed range given that
# stability arises with no active control. The weave steer-roll phase is almost
# aligned and the caster time constant is becoming very fast.
ax = model.plot_eigenvectors(hide_zeros=True, v=5.5, k9=128.0, c9=50.0)

# %%
times = np.linspace(0.0, 5.0, num=501)
ax = model.plot_mode_simulations(times, hide_zeros=True, v=5.5, k9=128.0,
                                 c9=50.0)

# %%
# High Speed, Unstable Capsize
# ----------------------------
#
# If the speed is increased further, the system becomes unstable due to the
# capsize mode. The weave steer and roll are mostly aligned in phase but the
# roll dominated capsize shows that the bicycle slowly falls over.
ax = model.plot_eigenvectors(hide_zeros=True, v=8.0, k9=128.0, c9=50.0)

# %%
ax = model.plot_mode_simulations(times, hide_zeros=True, v=8.0, k9=128.0,
                                 c9=50.0)
# %%
# Total Motion Simulation
# -----------------------
#
# The following plot shows the total motion with the same initial conditions
# used in [Meijaard2007]_ within the stable speed range at 4.6 m/s.
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

# %%
ax = model.plot_simulation(times, x0,
                           input_func=lambda t, x: [0.0, 0.0, 1.0, 0.0],
                           v=5.5, k9=128.0, c9=50.0)
