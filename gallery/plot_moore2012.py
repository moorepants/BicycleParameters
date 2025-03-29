r"""
=========================================
Carvallo-Whipple Model with Leaning Rider
=========================================

The Carvallo-Whipple bicycle model assumes that the rider is rigidly fixed to
the rear frame of the bicycle. Here the model is extended by including a rigid
body representing the rider's upper body (torso, arms, head) which can lean
relative to the rear frame through a single additional degree of freedom.

"""
import numpy as np

from bicycleparameters.parameter_dicts import (
    meijaard2007_browser_jasonlegs, moore2012riderlean_browser_jason)
from bicycleparameters.parameter_sets import (
    Meijaard2007ParameterSet, Moore2012RiderLeanParameterSet)
from bicycleparameters.models import Moore2012RiderLeanModel

# %%
par_set = Meijaard2007ParameterSet(meijaard2007_browser_jasonlegs, True)
par_set.to_parameterization('Moore2012').parameters

# %%
# Load Parameters
# ===============
# Linearize about a constant speed, so include the parameter :math:`v` for
# speed.
moore2012riderlean_browser_jason['v'] = 1.0
par_set = Moore2012RiderLeanParameterSet(moore2012riderlean_browser_jason)
par_set

# %%
# Linear Carvallo-Whipple with Rider Lean Model
# =============================================
#
model = Moore2012RiderLeanModel(par_set)

# %%
A, B = model.form_state_space_matrices(v=6.0)
A

# %%
B

# %%
# The root locus can be plotted as a function of any model parameter. This plot
# is most often used to show how stability is dependent on the speed
# parameter.The blue lines indicate the weave eigenmode, the green the capsize
# eigenmode, and the orange the caster eigenmode. Vertical dashed lines
# indicate speeds that are examined further below.
v = np.linspace(0.0, 10.0, num=401)
ax = model.plot_eigenvalue_parts(v=v, hide_zeros=True)

# %%
ax = model.plot_eigenvalue_parts(v=v, k9=128.0, c9=50.0, hide_zeros=True)

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
ax = model.plot_eigenvectors(v=0.0, k9=128.0, c9=50.0)

# %%
# The unstable eigenmodes dominate the motion and this results in the steer
# angle growing rapidly and the roll angle following suite, but slower.
times = np.linspace(0.0, 1.0)
ax = model.plot_mode_simulations(times, v=0.0, k9=128.0, c9=50.0)

# %%
# The total motion shows that the bicycle falls over, as expected.
x0 = np.zeros(13)
x0[3] = 0.01  # q4
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
ax = model.plot_eigenvectors(v=2.0, k9=128.0, c9=50.0)
# %%
times = np.linspace(0.0, 1.0)
ax = model.plot_mode_simulations(times, v=2.0, k9=128.0, c9=50.0)

# %%
# Stable Eigenmodes
# -----------------
#
# Increasing speed further shows that the weave eigenmode becomes stable,
# making all eigenmodes stable, and thus the total motion is stable. The speed
# regime where this is true is called the "self-stable" speed range given that
# stability arises with no active control. The weave steer-roll phase is almost
# aligned and the caster time constant is becoming very fast.
ax = model.plot_eigenvectors(v=5.0, k9=128.0, c9=50.0)
# %%
times = np.linspace(0.0, 5.0, num=501)
ax = model.plot_mode_simulations(times, v=5.0, k9=128.0, c9=50.0)

# %%
# High Speed, Unstable Capsize
# ----------------------------
#
# If the speed is increased further, the system becomes unstable due to the
# capsize mode. The weave steer and roll are mostly aligned in phase but the
# roll dominated capsize shows that the bicycle slowly falls over.
ax = model.plot_eigenvectors(v=8.0, k9=128.0, c9=50.0)

# %%
ax = model.plot_mode_simulations(times, v=8.0, k9=128.0, c9=50.0)

# %%
# Total Motion Simulation
# -----------------------
#
# The following plot shows the total motion with the same initial conditions
# used in [Meijaard2007]_ within the stable speed range at 4.6 m/s.
x0 = np.zeros(13)
x0[9] = 0.5  # u4
ax = model.plot_simulation(times, x0, v=6.0, k9=128.0, c9=50.0)

# %%
# A constant steer torque puts the model into a turn.
times = np.linspace(0.0, 10.0, num=1001)
ax = model.plot_simulation(times, x0,
                           input_func=lambda t, x: [0.0, 0.0, 1.0, 0.0],
                           v=6.0, k9=128.0, c9=50.0)
